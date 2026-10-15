"""Analyze frozen Binding predictions by seed and semantic event group."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
from statistics import mean, stdev

from src.grammars.generator import save_json, sha256

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/'results/runs/binding_formal_v1_20261015'
DATA = ROOT/'data/binding_formal_v1_20261015'
OUTPUT = ROOT/'results/aggregate/binding_formal_v1'
CONDITIONS = ('ro055_rm095', 'ro075_rm075', 'ro095_rm055')
ARCHITECTURES = ('rnn', 'lstm', 'transformer')
SEEDS = (11, 12, 13, 14, 15)
T4 = 2.7764451051977987


def interval(values):
    if len(values) != 5:
        raise ValueError('Five seed estimates required')
    avg = mean(values)
    half = T4*stdev(values)/math.sqrt(5)
    return {'mean': avg, 'sd': stdev(values), 'ci95': [avg-half, avg+half],
            'seed_values': dict(zip(map(str, SEEDS), values)),
            'unit': 'five trained runs; common test inventory'}


def row_metrics(row):
    p, q, y = row['p_query_agent'], row['oracle_p_query_agent'], row['label_query_is_agent']
    eps = 1e-15
    ptrue = min(1-eps, max(eps, p if y else 1-p))
    qtrue = q if y else 1-q
    return {'nll_regret': -math.log(ptrue)+math.log(qtrue),
            'posterior_mae': abs(p-q), 'accuracy': float((p >= .5) == bool(y))}


def grouped(rows, version, cell=None):
    groups = defaultdict(list)
    for row in rows:
        if row['version'] != version:
            continue
        if cell and (row['distance'], row['depth']) != cell:
            continue
        groups[row['event_group_id']].append(row_metrics(row))
    if not groups:
        raise ValueError('Empty analysis stratum')
    sizes = {len(values) for values in groups.values()}
    if len(sizes) != 1:
        raise ValueError('Unequal semantic-group sizes')
    return {key: {metric: mean(x[metric] for x in values) for metric in ('nll_regret', 'posterior_mae', 'accuracy')}
            for key, values in groups.items()}


def load_predictions(condition, arch, seed, split):
    path = RUNS/'runs'/condition/arch/f'seed{seed}'/f'{split}_predictions.jsonl'
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def cluster_interval(grouped_by_seed, metric, resamples=1000, seed=20261015):
    keys = sorted(grouped_by_seed[0])
    if any(set(values) != set(keys) for values in grouped_by_seed):
        raise ValueError('Semantic groups differ across training seeds')
    combined = {key: mean(values[key][metric] for values in grouped_by_seed) for key in keys}
    rng = random.Random(seed)
    boot = sorted(mean(combined[rng.choice(keys)] for _ in keys) for _ in range(resamples))
    return {'mean_fixed_trained_runs': mean(combined.values()),
            'ci95': [boot[24], boot[974]], 'event_groups': len(keys),
            'resamples': resamples,
            'unit': 'semantic event groups; trained runs held fixed'}


def summarize_stratum(rows_by_seed, version, cell=None):
    collections = [grouped(rows, version, cell) for rows in rows_by_seed]
    summary = {'event_groups': len(collections[0]), 'cells': cell,
               'metrics': {}}
    for metric in ('nll_regret', 'posterior_mae', 'accuracy'):
        seed_values = [mean(g[metric] for g in collection.values()) for collection in collections]
        summary['metrics'][metric] = {'training_seed': interval(seed_values),
                                      'test_group_conditional': cluster_interval(collections, metric)}
    return summary


def main(argv=None):
    global RUNS, DATA, OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=RUNS)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    RUNS, DATA, OUTPUT = args.runs, args.data, args.output
    summary = json.loads((RUNS/'summary.json').read_text())
    if summary['status'] != 'completed' or len(summary['runs']) != 45:
        raise ValueError('Formal Binding run plan incomplete')
    expected = {f'{c}/{a}/seed{s}' for c in CONDITIONS for a in ARCHITECTURES for s in SEEDS}
    if {run['run_id'] for run in summary['runs']} != expected:
        raise ValueError('Formal run IDs do not match the frozen plan')
    manifest_sha = sha256(DATA/'manifest.json')
    if summary['dataset_manifest_sha256'] != manifest_sha:
        raise ValueError('Data manifest changed after formal run')
    result = {'status': 'completed', 'scope': 'prespecified primary endpoints; descriptive seed and conditional group intervals',
              'dataset_manifest_sha256': manifest_sha, 'config_sha256': sha256(RUNS/'config.json'),
              'run_count': 45, 'groups': [], 'baselines': {}, 'failure_examples': []}
    certificates = json.loads((DATA/'information_certificate.json').read_text())
    for condition in CONDITIONS:
        certificate = certificates[condition]['test']
        result['baselines'][condition] = {
            'witness_pairs': certificate['same_suffix_opposite_truth_different_posterior_pairs'],
            'example': certificate['example'], 'test': certificate['baselines']}
        for arch in ARCHITECTURES:
            test_rows = [load_predictions(condition, arch, seed, 'test') for seed in SEEDS]
            ood_rows = [load_predictions(condition, arch, seed, 'ood') for seed in SEEDS]
            run_results = [next(run for run in summary['runs']
                                if run['run_id'] == f'{condition}/{arch}/seed{seed}') for seed in SEEDS]
            for run, rows in zip(run_results, test_rows):
                if len(rows) != 2400 or run['splits']['test']['checkpoint_reload_max_logit_error'] > 1e-6:
                    raise ValueError('Incomplete or unverified predictions')
            data = {'condition': condition, 'architecture': arch, 'seeds': list(SEEDS),
                    'test': {version: summarize_stratum(test_rows, version)
                             for version in ('binding', 'local')},
                    'ood': {version: summarize_stratum(ood_rows, version)
                            for version in ('binding', 'local')},
                    'matched_length': {f'{version}_d2_h{depth}': summarize_stratum(test_rows, version, (2, depth))
                                       for version in ('binding', 'local') for depth in (0, 1)}}
            # Paired local-versus-Binding regret for each trained run and event group.
            gaps = []
            for rows in test_rows:
                binding = grouped(rows, 'binding')
                local = grouped(rows, 'local')
                if set(binding) != set(local):
                    raise ValueError('Local/Binding event alignment failed')
                gaps.append({key: {'gap': binding[key]['nll_regret']-local[key]['nll_regret']}
                             for key in binding})
            gap_seed = [mean(value['gap'] for value in group.values()) for group in gaps]
            data['binding_minus_local_regret'] = {
                'training_seed': interval(gap_seed),
                'test_group_conditional': cluster_interval(gaps, 'gap')}
            result['groups'].append(data)
            if arch == 'transformer':
                by_group = defaultdict(list)
                for rows in test_rows:
                    for row in rows:
                        if row['version'] == 'binding':
                            by_group[row['event_group_id']].append((row_metrics(row)['nll_regret'], row['record_id']))
                for event_group, values in sorted(by_group.items(), key=lambda item: -mean(v[0] for v in item[1]))[:3]:
                    result['failure_examples'].append({'condition': condition, 'architecture': arch,
                       'event_group_id': event_group, 'mean_nll_regret': mean(v[0] for v in values),
                       'example_record_id': values[0][1]})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save_json(OUTPUT/'analysis.json', result)
    lines = ['# Formal RoleCue-Binding analysis', '',
             f"45/45 runs; data manifest SHA-256: `{manifest_sha}`.", '',
             'Five-seed intervals use trained runs. Group-bootstrap intervals resample unordered entity-pair/predicate groups while holding these five runs fixed; the two are not combined.', '',
             '| Condition | Model | Binding accuracy | Binding NLL regret | Local NLL regret | Binding minus local regret | Binding posterior MAE |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in result['groups']:
        metric = lambda version, name: row['test'][version]['metrics'][name]['training_seed']['mean']
        lines.append(f"| {row['condition']} | {row['architecture']} | {metric('binding','accuracy'):.3f} | "
                     f"{metric('binding','nll_regret'):.3f} | {metric('local','nll_regret'):.3f} | "
                     f"{row['binding_minus_local_regret']['training_seed']['mean']:.3f} | "
                     f"{metric('binding','posterior_mae'):.3f} |")
    lines += ['', 'The complete JSON contains each seed, both uncertainty sources, equal-length cells, OOD cells, and suffix-collision certificates.',
              'No architecture equivalence or inferential mechanism claim is made. The d4/h2 cut changes length and depth jointly.', '']
    (OUTPUT/'report.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
