"""Run a frozen boundary/smoke or multi-seed pilot suite with live progress."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import platform
from pathlib import Path
import random
import shutil
import sys
import time
import torch

from src.grammars.events import Event, derive_seed
from src.grammars.generator import make_record, save_json, sha256, write_jsonl
from src.grammars.validate import validate_record, statistics, validate_inventory
from src.models import build_model, parameter_count
from src.training.datamodule import read_records, load_vocabulary, verify_file, require_disjoint
from src.training.baselines import evaluate_baseline, fit_baselines
from src.training.train import train_one
from src.training.eval import paired_diagnostics, predict, prediction_rows, score


def prepare_boundaries(data_root, output, condition_ids):
    data_root, output = Path(data_root), Path(output)
    config = json.loads((data_root / 'config.json').read_text())
    sizes = config['stages']['smoke']
    inventory = {split: [] for split in sizes}
    for row in read_records(data_root / 'inventory/events.jsonl'):
        split = row['split']
        if len(inventory[split]) < sizes[split]:
            inventory[split].append(Event(row['predicate'], row['agent'], row['patient'], split))
    validate_inventory(inventory)
    reports = {}
    for condition_id in condition_ids:
        condition = next(c for c in config['boundary_conditions'] if c['id'] == condition_id)
        reports[condition_id] = {}
        for split, events in inventory.items():
            rows = [make_record(e, condition, config, 'boundary_training_v1') for e in events]
            random.Random(derive_seed(config['seeds']['export'], 'smoke', split)).shuffle(rows)
            for row in rows:
                validate_record(row, config)
            reports[condition_id][split] = statistics(rows, condition['r_o'], condition['r_m'])
            write_jsonl(output / condition_id / f'{split}.jsonl', rows)
    save_json(output / 'validation.json', reports)
    save_json(output / 'provenance.json', {
        'parent_dataset': str(data_root), 'parent_manifest_sha256': sha256(data_root / 'manifest.json'),
        'conditions': condition_ids, 'canonical_direction': config['canonical_direction'],
        'sizes': sizes, 'config': config,
        'note': 'Training/dev events come only from matching frozen inventory splits; test remains held out.'})


def passes_dev_gate(condition, metrics, gates):
    if condition == 'uninformative':
        passed = (gates['uninformative_accuracy_min'] <= metrics['accuracy'] <= gates['uninformative_accuracy_max']
                  and metrics['oracle_nll_regret'] <= gates['uninformative_nll_regret_max'])
    elif condition.startswith('perfect_'):
        passed = metrics['accuracy'] >= gates['perfect_accuracy_min']
    else:
        passed = (metrics['oracle_accuracy']-metrics['accuracy'] <= gates['smoke_accuracy_gap_max']
                  and metrics['oracle_nll_regret'] <= gates['smoke_nll_regret_max'])
    return bool(passed)


def run_plan(config):
    conditions = config.get('conditions') or config['smoke_conditions']
    return [(stage, condition, arch, seed)
            for stage, selected in (('boundary', config.get('boundary_conditions', [])),
                                    (config['stage'], conditions))
            for condition in selected for arch in config['models'] for seed in config['training_seeds']]


def training_settings(config, architecture):
    return dict(config['training'], **config.get('training_by_model', {}).get(architecture, {}))


def run_suite(config_path, output, resume=False):
    config_path, output = Path(config_path), Path(output)
    config = json.loads(config_path.read_text())
    if output.exists() and not resume:
        raise FileExistsError(f'Refusing to overwrite run: {output}')
    if resume and not output.exists():
        raise FileNotFoundError(f'Cannot resume missing run: {output}')
    if resume:
        frozen_config = json.loads((output / 'config.json').read_text())
        if frozen_config != config:
            raise ValueError('Resume config differs from the frozen run config')
    data_root = Path(config['dataset'])
    parent_manifest = json.loads((data_root / 'manifest.json').read_text())
    for name in ('config.json', 'inventory/events.jsonl', 'inventory/vocabulary.json'):
        verify_file(data_root / name, parent_manifest, data_root)
    conditions = config.get('conditions') or config['smoke_conditions']
    for condition in conditions:
        for split in ('train', 'dev', 'test', 'counterfactual'):
            verify_file(data_root / config['stage'] / condition / f'{split}.jsonl', parent_manifest, data_root)
    if not resume:
        output.mkdir(parents=True)
        save_json(output / 'config.json', config)
    source_root = Path(__file__).resolve().parents[2]
    benchmark = source_root / 'results/aggregate/smoke_device_benchmark.json'
    if benchmark.exists() and not resume:
        shutil.copy2(benchmark, output / 'device_benchmark.json')
    if not resume:
        for source in list((source_root / 'src').rglob('*.py')) + list((source_root / 'tests').glob('test_*.py')):
            if source.name.startswith('._'):
                continue
            target = output / 'source' / source.relative_to(source_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        shutil.copytree(source_root / 'configs', output / 'source/configs')
        shutil.copytree(source_root / 'scripts', output / 'source/scripts',
                        ignore=shutil.ignore_patterns('__pycache__', '._*'))
    torch.set_num_threads(config['cpu_threads'])
    torch.use_deterministic_algorithms(config['deterministic_algorithms'])
    device = torch.device(config['device'])
    vocabulary = load_vocabulary(data_root / 'inventory/vocabulary.json')
    counts = {arch: parameter_count(build_model(arch, len(vocabulary), values)) for arch, values in config['models'].items()}
    if max(counts.values())/min(counts.values()) > 1.05:
        raise ValueError('Model parameter budgets differ by more than 5%')
    environment = {
        'python': platform.python_version(), 'python_executable': sys.executable,
        'platform': platform.platform(), 'machine': platform.machine(),
        'processor': platform.processor(), 'device': str(device), 'torch': torch.__version__,
        'mps_available': torch.backends.mps.is_available(), 'cpu_threads': torch.get_num_threads(),
        'deterministic_algorithms': config['deterministic_algorithms'], 'parameter_counts': counts,
        'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
        'torch_build': torch.__config__.show()}
    if not resume:
        save_json(output / 'environment.json', environment)
        save_json(output / 'vocabulary.json', vocabulary)
    print(f'CONFIG FROZEN | device={device} | parameters={counts}', flush=True)
    if config.get('boundary_conditions') and not resume:
        prepare_boundaries(data_root, output / 'boundary_data', config['boundary_conditions'])
    runs = run_plan(config)
    results, start = [], time.monotonic()
    if resume:
        frozen_plan = json.loads((output / 'run_plan.json').read_text())
        if frozen_plan['runs'] != [list(run) for run in runs]:
            raise ValueError('Resume run plan differs from the frozen run plan')
        prior_summary = json.loads((output / 'summary.json').read_text())
        results = prior_summary.get('runs', [])
        completed_ids = {result['run_id'] for result in results}
        if len(completed_ids) != len(results):
            raise ValueError('Duplicate run IDs in resume summary')
        for run_id in completed_ids:
            evaluation = output / 'runs' / run_id / 'evaluation.json'
            if not evaluation.exists() or json.loads(evaluation.read_text()).get('run_id') != run_id:
                raise ValueError(f'Completed run lacks a valid evaluation: {run_id}')
        print(f'RESUME VERIFIED | completed={len(results)}/{len(runs)}', flush=True)
    else:
        completed_ids = set()
        save_json(output / 'run_plan.json', {'runs': runs, 'config_sha256': sha256(output / 'config.json'),
                                           'parent_manifest_sha256': sha256(data_root / 'manifest.json')})
    for index, (stage, condition, architecture, seed) in enumerate(runs, 1):
        run_id = f'{stage}/{condition}/{architecture}/seed{seed}'
        if run_id in completed_ids:
            print(f'SKIP VERIFIED {run_id}', flush=True)
            continue
        print(f'\n===== RUN {index}/{len(runs)}: {run_id} =====', flush=True)
        root = output / 'boundary_data' if stage == 'boundary' else data_root / config['stage']
        train_rows = read_records(root / condition / 'train.jsonl', 'train')
        dev_rows = read_records(root / condition / 'dev.jsonl', 'dev')
        expected_sizes = config.get('expected_sizes')
        if expected_sizes and (len(train_rows) != expected_sizes['train'] or len(dev_rows) != expected_sizes['dev']):
            raise ValueError('Dataset size differs from frozen run config')
        run_output = output / 'runs' / run_id

        def progress(message, **details):
            print(f'[{index}/{len(runs)} {condition}] {message}', flush=True)
            progress_state = {'status': 'training', 'run_index': index,
                       'runs_total': len(runs), 'run_id': run_id, 'message': message,
                       'elapsed_seconds': time.monotonic()-start, **details}
            temporary = output / 'progress.tmp.json'
            save_json(temporary, progress_state)
            temporary.replace(output / 'progress.json')

        model, training = train_one(architecture, train_rows, dev_rows, vocabulary,
                                    config['models'][architecture], training_settings(config, architecture), run_output,
                                    device, seed, progress)
        passed = passes_dev_gate(condition, training['best_dev'], config['dev_gates'])
        result = {'run_id': run_id, 'stage': stage, 'condition': condition, 'architecture': architecture,
                  'seed': seed, 'parameters': training['parameters'], 'best_epoch': training['best_epoch'],
                  'dev': training['best_dev'], 'dev_gate_passed': passed, 'train_seconds': training['seconds']}
        if not passed and config.get('stop_on_dev_gate_failure', True):
            results.append(result)
            save_json(output / 'summary.json', {'status': 'dev_gate_failed', 'runs': results})
            save_json(output / 'progress.json', {'status': 'dev_gate_failed', 'run_id': run_id})
            raise RuntimeError(f'Dev gate failed: {run_id}. No test-driven tuning; suite stopped.')
        # Test is opened for scoring only after checkpoint selection and the dev gate.
        test_rows = read_records(root / condition / 'test.jsonl', 'test')
        if expected_sizes and len(test_rows) != expected_sizes['test']:
            raise ValueError('Test size differs from frozen run config')
        require_disjoint(train_rows, dev_rows, test_rows)
        logits = predict(model, test_rows, vocabulary, device)
        result['test'] = score(logits, test_rows)
        write_jsonl(run_output / 'test_predictions.jsonl', prediction_rows(logits, test_rows))
        if stage != 'boundary':
            pairs = read_records(root / condition / 'counterfactual.jsonl', 'test')
            pair_logits = predict(model, pairs, vocabulary, device)
            result['counterfactual'] = score(pair_logits, pairs)
            result['paired_diagnostics'] = paired_diagnostics(pair_logits, pairs)
            write_jsonl(run_output / 'counterfactual_predictions.jsonl', prediction_rows(pair_logits, pairs))
        save_json(run_output / 'evaluation.json', result)
        results.append(result)
        save_json(output / 'summary.json', {'status': 'running', 'runs': results})
        print(f'{"PASS" if passed else "DEV_WARNING"} {run_id} | best_epoch={training["best_epoch"]} | '
              f'test_acc={result["test"]["accuracy"]:.4f} test_NLL={result["test"]["nll"]:.4f} '
              f'oracle_NLL_gap={result["test"]["oracle_nll_regret"]:+.4f}', flush=True)
    baseline_results = {}
    if config.get('evaluate_baselines'):
        print('===== TRAIN-ONLY BASELINES =====', flush=True)
        for condition in conditions:
            root = data_root / config['stage']
            train_rows = read_records(root / condition / 'train.jsonl', 'train')
            test_rows = read_records(root / condition / 'test.jsonl', 'test')
            pairs = read_records(root / condition / 'counterfactual.jsonl', 'test')
            require_disjoint(train_rows, test_rows)
            baseline_results[condition] = {}
            for baseline in fit_baselines(train_rows, vocabulary, config.get('bag_of_words', {})):
                name = baseline.report()['name']
                print(f'BASELINE {condition}/{name}', flush=True)
                evaluated = evaluate_baseline(baseline, test_rows, pairs)
                baseline_output = output / 'baselines' / condition / name
                test_predictions = evaluated.pop('test_prediction_rows')
                cf_predictions = evaluated.pop('counterfactual_prediction_rows')
                save_json(baseline_output / 'evaluation.json', evaluated)
                write_jsonl(baseline_output / 'test_predictions.jsonl', test_predictions)
                write_jsonl(baseline_output / 'counterfactual_predictions.jsonl', cf_predictions)
                baseline_results[condition][name] = evaluated
        save_json(output / 'baselines' / 'summary.json', baseline_results)
    summary = {'status': 'completed', 'runs': results, 'baselines': baseline_results,
               'total_seconds': time.monotonic()-start,
               'training_seeds': config['training_seeds'],
               'scope': config.get('scope', 'single-seed engineering smoke, not confirmatory comparison')}
    save_json(output / 'summary.json', summary)
    save_json(output / 'progress.json', {'status': 'completed', 'runs_completed': len(results),
                                       'elapsed_seconds': time.monotonic()-start})
    files = {str(p.relative_to(output)): {'sha256': sha256(p), 'bytes': p.stat().st_size}
             for p in sorted(output.rglob('*')) if p.is_file()
             and not any(part.startswith('._') for part in p.relative_to(output).parts)}
    save_json(output / 'manifest.json', {'status': 'completed', 'files': files,
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'parent_manifest_sha256': sha256(data_root / 'manifest.json')})
    print(f'COMPLETE: {len(results)} runs, {summary["total_seconds"]:.1f}s, {output}', flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/train_smoke_v1.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run_suite(args.config, args.output, args.resume)
