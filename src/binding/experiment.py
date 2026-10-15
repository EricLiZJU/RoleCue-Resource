"""Run the frozen Binding protocol; every test is evaluated after dev selection."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import platform
from pathlib import Path
import shutil
import sys
import time

import torch

from src.binding.task import oracle_from_tokens
from src.grammars.generator import save_json, sha256, write_jsonl
from src.models import build_model, parameter_count
from src.training.datamodule import read_records, require_disjoint
from src.training.eval import predict, score
from src.training.train import train_one


def _prediction_rows(logits, rows):
    probabilities = logits.sigmoid().double().cpu().tolist()
    for p, row in zip(probabilities, rows):
        yield {'record_id': row['record_id'], 'event_id': row['event_id'],
               'event_group_id': row['event_group_id'], 'condition': row['condition'],
               'version': row['version'], 'distance': row['distance'], 'depth': row['depth'],
               'label_query_is_agent': row['label_query_is_agent'],
               'oracle_p_query_agent': row['oracle_p_query_agent'], 'p_query_agent': p,
               'cue_conflict': row['cue_conflict']}


def _by_cell(logits, rows):
    result = {}
    for version in ('binding', 'local'):
        for distance, depth in sorted({(r['distance'], r['depth']) for r in rows}):
            indices = [i for i, row in enumerate(rows)
                       if row['version'] == version and (row['distance'], row['depth']) == (distance, depth)]
            if indices:
                result[f'{version}_d{distance}_h{depth}'] = score(logits[indices], [rows[i] for i in indices])
    return result


def _verify_dataset(root):
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['status'] != 'validated':
        raise ValueError('Binding dataset has not passed validation')
    for name, info in manifest['files'].items():
        if sha256(root/name) != info['sha256']:
            raise ValueError(f'Dataset hash mismatch: {name}')
    certificate = json.loads((root/'information_certificate.json').read_text())
    for condition in certificate.values():
        for split in ('dev', 'test', 'ood'):
            if condition[split]['same_suffix_opposite_truth_different_posterior_pairs'] < 1:
                raise ValueError('Missing local-information witness')
    return sha256(root/'manifest.json')


def run(config_path, output, resume=False):
    config_path, output = Path(config_path), Path(output)
    config = json.loads(config_path.read_text())
    data_root = Path(config['dataset'])
    dataset_sha = _verify_dataset(data_root)
    if output.exists() and not resume:
        raise FileExistsError(output)
    if resume:
        if json.loads((output/'config.json').read_text()) != config:
            raise ValueError('Resume configuration differs from frozen configuration')
    else:
        output.mkdir(parents=True)
        save_json(output/'config.json', config)
        save_json(output/'provenance.json', {
            'dataset_manifest_sha256': dataset_sha,
            'config_source_sha256': sha256(config_path),
            'python': platform.python_version(), 'torch': torch.__version__,
            'device': config['device'], 'platform': platform.platform(),
            'machine': platform.machine(), 'python_executable': sys.executable,
            'packages': {p.metadata['Name']: p.version for p in importlib.metadata.distributions()},
            'created_utc': datetime.now(timezone.utc).isoformat()})
        source_root = Path(__file__).resolve().parents[2]
        for relative in ('src/binding/task.py', 'src/binding/build.py', 'src/binding/experiment.py',
                         'src/training/train.py', 'src/training/eval.py',
                         'src/grammars/linearizer.py', 'src/baselines/bayes_oracle.py'):
            dest = output/'source'/relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_root/relative, dest)
    torch.set_num_threads(config['cpu_threads'])
    torch.use_deterministic_algorithms(config['deterministic_algorithms'])
    vocabulary_doc = json.loads((data_root/'vocabulary.json').read_text())
    vocabulary = {token: i for i, token in enumerate(vocabulary_doc['entities']+
                    vocabulary_doc['predicates']+vocabulary_doc['special_tokens'])}
    if len(vocabulary) != sum(len(vocabulary_doc[key]) for key in ('entities', 'predicates', 'special_tokens')):
        raise ValueError('Vocabulary has duplicate tokens')
    counts = {arch: parameter_count(build_model(arch, len(vocabulary), settings))
              for arch, settings in config['models'].items()}
    save_json(output/'parameter_counts.json', counts)
    plan = [(condition, architecture, seed) for condition in config['conditions']
            for architecture in config['models'] for seed in config['training_seeds']]
    if not resume:
        save_json(output/'run_plan.json', {'runs': plan, 'dataset_manifest_sha256': dataset_sha,
                                         'config_sha256': sha256(output/'config.json')})
    elif json.loads((output/'run_plan.json').read_text())['runs'] != [list(p) for p in plan]:
        raise ValueError('Resume plan mismatch')
    started = time.monotonic()
    summary = {'status': 'running', 'runs': [], 'planned_runs': len(plan),
               'dataset_manifest_sha256': dataset_sha,
               'scope': config['scope'], 'parameter_counts': counts}
    if resume:
        summary = json.loads((output/'summary.json').read_text())
    done = {r['run_id'] for r in summary['runs']}
    device = torch.device(config['device'])
    for index, (condition, arch, seed) in enumerate(plan, 1):
        run_id = f'{condition}/{arch}/seed{seed}'
        if run_id in done:
            if not (output/'runs'/run_id/'evaluation.json').exists():
                raise ValueError(f'Incomplete completed run {run_id}')
            continue
        root = data_root/condition
        train = read_records(root/'train.jsonl', 'train')
        dev = read_records(root/'dev.jsonl', 'dev')
        test = read_records(root/'test.jsonl', 'test')
        ood = read_records(root/'ood.jsonl', 'ood')
        require_disjoint(train, dev, test, ood)
        for row in test[:5]+ood[:5]:
            if abs(oracle_from_tokens(row['tokens'], row['r_o'], row['r_m'])-row['oracle_p_query_agent']) > 1e-12:
                raise ValueError('Frozen oracle mismatch')
        run_root = output/'runs'/run_id
        settings = dict(config['training'], **config.get('training_by_model', {}).get(arch, {}))
        def progress(message, **details):
            print(f'[{index}/{len(plan)} {run_id}] {message}', flush=True)
            save_json(output/'progress.json', {'status': 'training', 'run_id': run_id,
                      'run_index': index, 'planned_runs': len(plan), 'message': message,
                      'elapsed_seconds': time.monotonic()-started, **details})
        model, training = train_one(arch, train, dev, vocabulary,
                                    config['models'][arch], settings, run_root,
                                    device, seed, progress)
        evaluation = {'run_id': run_id, 'condition': condition, 'architecture': arch,
                      'seed': seed, 'best_epoch': training['best_epoch'],
                      'parameters': training['parameters'], 'train_seconds': training['seconds'],
                      'dev': training['best_dev'], 'splits': {}}
        for split, rows in (('test', test), ('ood', ood)):
            logits = predict(model, rows, vocabulary, device)
            evaluation['splits'][split] = {'all': score(logits, rows), 'cells': _by_cell(logits, rows)}
            write_jsonl(run_root/f'{split}_predictions.jsonl', _prediction_rows(logits, rows))
            # A fresh model instance must reproduce the saved probabilities.
            checkpoint = torch.load(run_root/'best.pt', map_location='cpu', weights_only=False)
            reloaded = build_model(arch, len(vocabulary), config['models'][arch]).to(device)
            reloaded.load_state_dict(checkpoint['model_state'])
            restored = predict(reloaded, rows, vocabulary, device)
            diff = (restored-logits).abs().max().item()
            if diff > 1e-6:
                raise AssertionError(f'Checkpoint reload discrepancy: {diff}')
            evaluation['splits'][split]['checkpoint_reload_max_logit_error'] = diff
        save_json(run_root/'evaluation.json', evaluation)
        summary['runs'].append(evaluation)
        save_json(output/'summary.json', summary)
        print(f'COMPLETE {run_id} | test regret={evaluation["splits"]["test"]["all"]["oracle_nll_regret"]:+.4f}',
              flush=True)
    summary['status'] = 'completed'
    summary['elapsed_seconds_this_invocation'] = time.monotonic()-started
    save_json(output/'summary.json', summary)
    save_json(output/'progress.json', {'status': 'completed', 'runs_completed': len(summary['runs'])})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.config, args.output, args.resume)
