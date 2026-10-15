"""Standard-library RoleCue evaluator for external probability predictions."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def _records(path):
    with Path(path).open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError('Evaluation dataset is empty')
    ids = [row['record_id'] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Evaluation dataset has duplicate record_id values')
    return rows


def _prediction_mapping(path):
    path = Path(path)
    if path.suffix == '.csv':
        with path.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = _records(path)
    prediction = {}
    for row in rows:
        record_id = row['record_id']
        if record_id in prediction:
            raise ValueError(f'Duplicate prediction: {record_id}')
        keys = [key for key in ('p_query_agent', 'p_first_agent', 'probability') if key in row]
        if len(keys) != 1:
            raise ValueError('Use exactly one probability field per prediction')
        p = float(row[keys[0]])
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError(f'Invalid probability for {record_id}')
        prediction[record_id] = p
    return prediction


def _truth_oracle(row):
    if row.get('tokenizer') == 'rolecue_binding_v1':
        return row['label_query_is_agent'], row['oracle_p_query_agent'], 'binding'
    return row['label_first_is_agent'], row['oracle']['evaluation_p_first_agent'], 'core'


def _metrics(rows, predictions):
    n = len(rows)
    nll = oracle_nll = brier = mae = correct = 0.
    eps = 1e-15
    for row in rows:
        y, q, _ = _truth_oracle(row)
        p = predictions[row['record_id']]
        p_safe = min(1-eps, max(eps, p))
        q_true = q if y else 1-q
        if not 0 < q_true <= 1:
            raise ValueError('Oracle assigns zero probability to observed truth')
        nll -= y*math.log(p_safe)+(1-y)*math.log1p(-p_safe)
        oracle_nll -= math.log(q_true)
        brier += (p-y)**2
        mae += abs(p-q)
        correct += (p >= .5) == bool(y)
    return {'n': n, 'accuracy': correct/n, 'nll': nll/n,
            'oracle_nll': oracle_nll/n, 'oracle_nll_regret': (nll-oracle_nll)/n,
            'brier': brier/n, 'oracle_probability_mae': mae/n}


def evaluate(dataset, prediction_file):
    rows = _records(dataset)
    predictions = _prediction_mapping(prediction_file)
    ids = {row['record_id'] for row in rows}
    if set(predictions) != ids:
        raise ValueError(f'Prediction IDs differ: missing={len(ids-set(predictions))}, '
                         f'extra={len(set(predictions)-ids)}')
    modules = {_truth_oracle(row)[2] for row in rows}
    if len(modules) != 1:
        raise ValueError('Do not mix Core and Binding records in one evaluation file')
    groups = {}
    if 'binding' in modules:
        for version in ('binding', 'local'):
            selected = [row for row in rows if row['version'] == version]
            groups[version] = _metrics(selected, predictions)
            for distance, depth in sorted({(r['distance'], r['depth']) for r in selected}):
                cell = [row for row in selected if (row['distance'], row['depth']) == (distance, depth)]
                groups[f'{version}_d{distance}_h{depth}'] = _metrics(cell, predictions)
    else:
        for name, predicate in (('conflict', True), ('aligned', False)):
            selected = [row for row in rows if row['cue_conflict'] == predicate]
            if selected:
                groups[name] = _metrics(selected, predictions)
    return {'module': next(iter(modules)), 'dataset': str(Path(dataset)),
            'predictions': str(Path(prediction_file)), 'all': _metrics(rows, predictions),
            'groups': groups}


def verify_manifest(root):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    for name, info in manifest['files'].items():
        path = root/name
        if hashlib.sha256(path.read_bytes()).hexdigest() != info['sha256']:
            raise ValueError(f'Hash mismatch: {name}')
    return {'status': 'verified', 'files': len(manifest['files']), 'root': str(root)}


def verify_release_manifest(root):
    root = Path(root)
    manifest = json.loads((root/'release_manifest.json').read_text())
    for name, info in manifest['files'].items():
        path = root/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != info['sha256']:
            raise ValueError(f'Release file missing or hash mismatch: {name}')
    return {'status': 'verified', 'files': len(manifest['files']), 'root': str(root),
            'version': manifest['version']}


def make_example(dataset, output):
    rows = _records(dataset)
    with Path(output).open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(('record_id', 'probability'))
        for row in rows:
            _, q, _ = _truth_oracle(row)
            writer.writerow((row['record_id'], format(q, '.17g')))
    return len(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    evaluate_parser = commands.add_parser('evaluate', help='Score an external JSONL/CSV prediction file')
    evaluate_parser.add_argument('--dataset', required=True)
    evaluate_parser.add_argument('--predictions', required=True)
    evaluate_parser.add_argument('--output')
    verify_parser = commands.add_parser('verify', help='Check a frozen data manifest')
    verify_parser.add_argument('--root', required=True)
    release_parser = commands.add_parser('verify-release', help='Check every file in the release manifest')
    release_parser.add_argument('--root', required=True)
    example_parser = commands.add_parser('example-predictions', help='Write oracle example in public input format')
    example_parser.add_argument('--dataset', required=True)
    example_parser.add_argument('--output', required=True)
    figure_parser = commands.add_parser('figure', help='Rebuild the Core reliability-grid main figure')
    figure_parser.add_argument('--snapshots', required=True)
    figure_parser.add_argument('--output', required=True)
    binding_figure_parser = commands.add_parser('binding-figure', help='Rebuild the Binding seed-level diagnostic figure')
    binding_figure_parser.add_argument('--analysis', required=True)
    binding_figure_parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if args.command == 'evaluate':
        report = evaluate(args.dataset, args.predictions)
        content = json.dumps(report, indent=2, allow_nan=False)+'\n'
        if args.output:
            Path(args.output).write_text(content)
        else:
            print(content, end='')
    elif args.command == 'verify':
        print(json.dumps(verify_manifest(args.root), indent=2))
    elif args.command == 'verify-release':
        print(json.dumps(verify_release_manifest(args.root), indent=2))
    elif args.command == 'example-predictions':
        print(f'Wrote {make_example(args.dataset, args.output)} predictions to {args.output}')
    elif args.command == 'figure':
        from src.release.figure import reliability_grid
        result = reliability_grid(args.snapshots, args.output)
        print(json.dumps(result, indent=2))
    else:
        from src.release.figure import binding_diagnostic
        result = binding_diagnostic(args.analysis, args.output)
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
