"""Equal-budget, dev-only hyperparameter selection on the frozen pilot data."""
import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import shutil
import statistics
import time
import torch

from src.grammars.generator import save_json, sha256
from src.models import build_model, parameter_count
from src.training.datamodule import load_vocabulary, read_records, verify_file
from src.training.train import train_one


def candidates(config):
    keys = sorted(config['search_grid'])
    return [dict(zip(keys, values)) for values in itertools.product(*(config['search_grid'][k] for k in keys))]


def selection_table(results, architectures, variants):
    table, winners = [], {}
    for architecture in architectures:
        rows = []
        for index, variant in enumerate(variants):
            selected = [r['best_dev']['oracle_nll_regret'] for r in results
                        if r['architecture'] == architecture and r['candidate_index'] == index]
            if not selected:
                raise ValueError('Incomplete tuning result grid')
            rows.append({'architecture': architecture, 'candidate_index': index, 'hyperparameters': variant,
                         'mean_dev_oracle_nll_regret': statistics.mean(selected),
                         'sd_dev_oracle_nll_regret': statistics.stdev(selected), 'runs': len(selected)})
        # Frozen tie break: grid order after the primary mean dev-regret endpoint.
        winner = min(rows, key=lambda r: (r['mean_dev_oracle_nll_regret'], r['candidate_index']))
        table.extend(rows)
        winners[architecture] = winner
    return table, winners


def tune(config_path, output, resume=False):
    config_path, output = Path(config_path), Path(output)
    config = json.loads(config_path.read_text())
    data_root = Path(config['dataset'])
    manifest = json.loads((data_root/'manifest.json').read_text())
    # Deliberately fingerprint train/dev only. This process never opens a test or counterfactual file.
    for name in ('config.json', 'inventory/events.jsonl', 'inventory/vocabulary.json'):
        verify_file(data_root/name, manifest, data_root)
    for condition in config['conditions']:
        for split in ('train', 'dev'):
            verify_file(data_root/config['stage']/condition/f'{split}.jsonl', manifest, data_root)
    frozen_hash = sha256(config_path)
    if output.exists():
        if not resume:
            raise FileExistsError(f'Refusing to overwrite tuning run: {output}')
        if json.loads((output/'config.json').read_text()) != config:
            raise ValueError('Resume config differs from frozen tuning config')
    else:
        output.mkdir(parents=True)
        save_json(output/'config.json', config)
        source_root = Path(__file__).resolve().parents[2]
        for source in list((source_root/'src').rglob('*.py')) + list((source_root/'tests').glob('test_*.py')):
            if not source.name.startswith('._'):
                target = output/'source'/source.relative_to(source_root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        shutil.copytree(source_root/'configs', output/'source/configs')
    torch.set_num_threads(config['cpu_threads'])
    torch.use_deterministic_algorithms(config['deterministic_algorithms'])
    device = torch.device(config['device'])
    vocabulary = load_vocabulary(data_root/'inventory/vocabulary.json')
    variants = candidates(config)
    plan = [(a, i, hp, c, s) for a in config['models'] for i, hp in enumerate(variants)
            for c in config['conditions'] for s in config['training_seeds']]
    save_json(output/'run_plan.json', {'runs': [{'architecture': a, 'candidate_index': i,
              'hyperparameters': hp, 'condition': c, 'seed': s} for a,i,hp,c,s in plan],
              'config_sha256': frozen_hash, 'test_files_opened': False})
    results, started = [], time.monotonic()
    for index, (architecture, candidate_index, hp, condition, seed) in enumerate(plan, 1):
        run_id = f'{architecture}/candidate{candidate_index}/{condition}/seed{seed}'
        run_output = output/'runs'/run_id
        if resume and (run_output/'training.json').exists():
            training = json.loads((run_output/'training.json').read_text())
            print(f'RESUME {index}/{len(plan)} already complete: {run_id}', flush=True)
        else:
            train_rows = read_records(data_root/config['stage']/condition/'train.jsonl', 'train')
            dev_rows = read_records(data_root/config['stage']/condition/'dev.jsonl', 'dev')
            if [len(train_rows), len(dev_rows)] != [config['expected_sizes']['train'], config['expected_sizes']['dev']]:
                raise ValueError('Pilot tuning data size differs from config')
            settings = dict(config['training'], **hp)
            print(f'===== TUNE {index}/{len(plan)} {run_id} {hp} =====', flush=True)
            def progress(message, **details):
                print(f'[{index}/{len(plan)}] {message}', flush=True)
                save_json(output/'progress.tmp.json', {'status': 'training', 'run_index': index,
                          'runs_total': len(plan), 'run_id': run_id, 'message': message,
                          'elapsed_seconds': time.monotonic()-started, **details})
                (output/'progress.tmp.json').replace(output/'progress.json')
            _, training = train_one(architecture, train_rows, dev_rows, vocabulary,
                                    config['models'][architecture], settings, run_output, device, seed, progress)
        results.append({'run_id': run_id, 'architecture': architecture, 'candidate_index': candidate_index,
                        'hyperparameters': hp, 'condition': condition, 'seed': seed,
                        'best_epoch': training['best_epoch'], 'best_dev': training['best_dev'],
                        'last_dev': training['history'][-1]['dev'], 'parameters': training['parameters']})
        save_json(output/'summary.json', {'status': 'running', 'runs': results,
                  'scope': 'dev-only equal-budget tuning; no test or counterfactual file opened'})
    table, winners = selection_table(results, list(config['models']), variants)
    selected = {'status': 'frozen', 'selected_utc': datetime.now(timezone.utc).isoformat(),
                'selection_endpoint': 'mean dev oracle NLL regret over all conditions and tuning seeds',
                'tie_break': 'candidate grid order', 'test_files_opened': False,
                'data_manifest_sha256': sha256(data_root/'manifest.json'),
                'tuning_config_sha256': frozen_hash, 'table': table, 'winners': winners,
                'models': config['models'], 'base_training': config['training']}
    save_json(output/'selected_config.json', selected)
    save_json(output/'summary.json', {'status': 'completed', 'runs': results,
              'scope': 'dev-only equal-budget tuning; no test or counterfactual file opened',
              'total_seconds': time.monotonic()-started, 'selected_config': 'selected_config.json'})
    save_json(output/'progress.json', {'status': 'completed', 'runs_completed': len(results),
              'elapsed_seconds': time.monotonic()-started})
    files = {str(p.relative_to(output)): {'sha256': sha256(p), 'bytes': p.stat().st_size}
             for p in sorted(output.rglob('*')) if p.is_file() and not any(x.startswith('._') for x in p.relative_to(output).parts)}
    save_json(output/'manifest.json', {'status': 'completed', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'files': files, 'test_files_opened': False, 'parent_manifest_sha256': sha256(data_root/'manifest.json')})
    print(f'TUNING COMPLETE: {len(results)} dev-only runs | {output}', flush=True)
    return selected


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    tune(args.config, args.output, args.resume)
