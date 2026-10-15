"""Pack and independently rescore every formal Core main-run prediction.

The ZIP stores one compact IID CSV and one entity-aligned intervention CSV per
run. These are lossless for the paper's main scoring endpoints, but deliberately
exclude repeated token strings and other non-scoring JSONL metadata. Original
file hashes and row counts are retained for provenance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import zipfile

SCHEMA = 'rolecue.core-main-evidence.v1'
VARIANTS = ('reference', 'order_swap', 'marker_flip', 'both')
IID_HEADER = ('event_id', 'y', 'p_first', 'q_first', 'cue_conflict')
CF_HEADER = ('event_id', 'variant', 'p_aligned', 'q_aligned')
METRICS = ('accuracy', 'nll', 'oracle_nll_regret', 'oracle_probability_mae',
           'conflict_oracle_probability_mae', 'conflict_signed_bias',
           'paired_order_effect', 'paired_marking_effect',
           'paired_order_deviation', 'paired_marking_deviation')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def member_sha(archive: zipfile.ZipFile, name: str) -> str:
    h = hashlib.sha256()
    with archive.open(name) as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def event_group(event_id: str) -> str:
    predicate, pair = event_id.split(':', 1)
    left, right = pair.split('>', 1)
    return ':'.join((predicate, *sorted((left, right))))


def pack(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    source_manifest = json.loads((source / 'manifest.json').read_text())
    summary = json.loads((source / 'summary.json').read_text())
    if summary['status'] != 'completed' or len(summary['runs']) != 135:
        raise ValueError('Expected all 135 frozen Core main runs')
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {'schema': SCHEMA, 'source_manifest_sha256': sha(source / 'manifest.json'),
                'source_summary_sha256': sha(source / 'summary.json'),
                'runs': {}, 'columns': {'iid': IID_HEADER, 'counterfactual': CF_HEADER},
                'note': 'Derived scoring fields; token strings, checkpoint tensors and repeated metadata omitted.'}
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6,
                         allowZip64=True) as archive:
        for run in sorted(summary['runs'], key=lambda r: r['run_id']):
            rid = run['run_id']
            entries = {}
            for kind, filename, header in (
                    ('iid', 'test_predictions.jsonl', IID_HEADER),
                    ('counterfactual', 'counterfactual_predictions.jsonl', CF_HEADER)):
                relative = f'runs/{rid}/{filename}'
                original = source / relative
                expected = source_manifest['files'][relative]
                if original.stat().st_size != expected['bytes'] or sha(original) != expected['sha256']:
                    raise ValueError(f'Frozen source hash mismatch: {relative}')
                member = f'runs/{rid}/{kind}.csv'
                count = 0
                with archive.open(member, 'w', force_zip64=True) as raw:
                    with io.TextIOWrapper(raw, encoding='utf-8', newline='') as text:
                        writer = csv.writer(text)
                        writer.writerow(header)
                        with original.open(encoding='utf-8') as records:
                            for line in records:
                                row = json.loads(line)
                                if kind == 'iid':
                                    if row['variant'] is not None or row['pair_id'] is not None:
                                        raise ValueError(f'Unexpected IID variant in {rid}')
                                    writer.writerow((row['event_id'], row['target_first_agent'],
                                                     repr(row['p_first_agent']),
                                                     repr(row['oracle']['evaluation_p_first_agent']),
                                                     int(row['cue_conflict'])))
                                else:
                                    agent = row['event_id'].split(':', 1)[1].split('>', 1)[0]
                                    aligned_first = row['candidate_order'][0] == agent
                                    p_first = row['p_first_agent']
                                    p_aligned = p_first if aligned_first else 1 - p_first
                                    if abs(p_aligned - row['p_alignment_entity_agent']) > 1e-12:
                                        raise ValueError(f'Entity alignment mismatch in {rid}')
                                    q_first = row['oracle']['training_p_first_agent']
                                    q_aligned = q_first if aligned_first else 1 - q_first
                                    if row['variant'] not in VARIANTS:
                                        raise ValueError(f'Unexpected intervention variant in {rid}')
                                    writer.writerow((row['event_id'], row['variant'],
                                                     repr(row['p_alignment_entity_agent']), repr(q_aligned)))
                                count += 1
                expected_count = 20000 if kind == 'iid' else 80000
                if count != expected_count:
                    raise ValueError(f'Wrong {kind} count in {rid}: {count}')
                entries[kind] = {'member': member, 'rows': count,
                                 'source_sha256': expected['sha256'],
                                 'packed_sha256': member_sha(archive, member)}
            manifest['runs'][rid] = entries
            print(f'packed {len(manifest["runs"])}/135 {rid}', flush=True)
        archive.writestr('manifest.json', json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    print(json.dumps({'archive': str(output), 'bytes': output.stat().st_size,
                      'sha256': sha(output), 'runs': len(manifest['runs'])}, indent=2))


def _read_csv(archive: zipfile.ZipFile, name: str, header):
    with archive.open(name) as raw:
        with io.TextIOWrapper(raw, encoding='utf-8', newline='') as text:
            reader = csv.DictReader(text)
            if tuple(reader.fieldnames or ()) != tuple(header):
                raise ValueError(f'Unexpected columns: {name}')
            yield from reader


def score_run(archive: zipfile.ZipFile, entries: dict) -> dict:
    iid = entries['iid']
    n = 0
    correct = 0
    nll_terms, oracle_terms, mae_terms = [], [], []
    conflict_abs, conflict_bias = [], []
    groups = set()
    eps = 2.220446049250313e-16
    for row in _read_csv(archive, iid['member'], IID_HEADER):
        y = int(row['y']); p = float(row['p_first']); q = float(row['q_first'])
        conflict = int(row['cue_conflict'])
        if y not in (0, 1) or conflict not in (0, 1) or not 0 <= p <= 1 or not 0 < q < 1:
            raise ValueError('Invalid IID row')
        groups.add(event_group(row['event_id']))
        n += 1; correct += int((p >= .5) == bool(y))
        ptrue = min(1-eps, max(eps, p if y else 1-p))
        qtrue = q if y else 1-q
        nll_terms.append(-math.log(ptrue)); oracle_terms.append(-math.log(qtrue))
        mae_terms.append(abs(p-q))
        if conflict:
            conflict_abs.append(abs(p-q)); conflict_bias.append(p-q)
    if n != iid['rows'] or n != 20000 or len(groups) != 10000:
        raise ValueError('IID count or semantic grouping differs')
    means = {'accuracy': correct/n, 'nll': math.fsum(nll_terms)/n,
             'oracle_nll_regret': math.fsum(nll_terms)/n - math.fsum(oracle_terms)/n,
             'oracle_probability_mae': math.fsum(mae_terms)/n,
             'conflict_oracle_probability_mae': math.fsum(conflict_abs)/len(conflict_abs),
             'conflict_signed_bias': math.fsum(conflict_bias)/len(conflict_bias)}
    cf = entries['counterfactual']
    pairs = {}
    cf_count = 0
    for row in _read_csv(archive, cf['member'], CF_HEADER):
        event = row['event_id']; variant = row['variant']
        p = float(row['p_aligned']); q = float(row['q_aligned'])
        if variant not in VARIANTS or not 0 <= p <= 1 or not 0 <= q <= 1:
            raise ValueError('Invalid counterfactual row')
        group = pairs.setdefault(event, {})
        if variant in group:
            raise ValueError(f'Duplicate pair member: {event}/{variant}')
        group[variant] = (p, q)
        cf_count += 1
    if cf_count != cf['rows'] or cf_count != 80000 or len(pairs) != 20000:
        raise ValueError('Counterfactual count differs')
    order, marking, oracle_order, oracle_marking = [], [], [], []
    for members in pairs.values():
        if set(members) != set(VARIANTS):
            raise ValueError('Incomplete intervention quartet')
        ref, swap, flip = (members[k] for k in VARIANTS[:3])
        order.append(swap[0]-ref[0]); marking.append(flip[0]-ref[0])
        oracle_order.append(swap[1]-ref[1]); oracle_marking.append(flip[1]-ref[1])
    means.update({'paired_order_effect': math.fsum(order)/len(order),
                  'paired_marking_effect': math.fsum(marking)/len(marking),
                  'paired_order_deviation': math.fsum(order)/len(order) - math.fsum(oracle_order)/len(oracle_order),
                  'paired_marking_deviation': math.fsum(marking)/len(marking) - math.fsum(oracle_marking)/len(oracle_marking),
                  'n_iid': n, 'n_conflict': len(conflict_abs), 'n_pairs': len(pairs)})
    return means


def verify(archive_path: Path, summary_path: Path, aggregate_path: Path, output: Path | None) -> dict:
    summary = json.loads(summary_path.read_text())
    aggregate = json.loads(aggregate_path.read_text())
    if len(summary['runs']) != 135:
        raise ValueError('Frozen summary does not contain 135 runs')
    expected_runs = {r['run_id']: r for r in summary['runs']}
    by_cell = {}
    max_error = 0.0
    calculated = {}
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError('Damaged ZIP member')
        manifest = json.loads(archive.read('manifest.json'))
        if manifest['schema'] != SCHEMA or set(manifest['runs']) != set(expected_runs):
            raise ValueError('Run inventory mismatch')
        for rid, entries in sorted(manifest['runs'].items()):
            for entry in entries.values():
                if member_sha(archive, entry['member']) != entry['packed_sha256']:
                    raise ValueError(f'Packed member hash mismatch: {entry["member"]}')
            actual = score_run(archive, entries)
            reference = expected_runs[rid]
            checks = {'accuracy': reference['test']['accuracy'],
                      'nll': reference['test']['nll'],
                      'oracle_nll_regret': reference['test']['oracle_nll_regret'],
                      'oracle_probability_mae': reference['test']['oracle_probability_mae'],
                      'conflict_oracle_probability_mae': reference['test']['conflict']['oracle_probability_mae'],
                      'conflict_signed_bias': reference['test']['conflict']['oracle_probability_signed_bias'],
                      'paired_order_effect': reference['paired_diagnostics']['anchored_order_effect'],
                      'paired_marking_effect': reference['paired_diagnostics']['anchored_marking_effect'],
                      'paired_order_deviation': reference['paired_diagnostics']['anchored_order_effect'] - reference['paired_diagnostics']['oracle_anchored_order_effect'],
                      'paired_marking_deviation': reference['paired_diagnostics']['anchored_marking_effect'] - reference['paired_diagnostics']['oracle_anchored_marking_effect']}
            for metric, expected in checks.items():
                error = abs(actual[metric]-expected)
                max_error = max(max_error, error)
                if error > 2e-6:
                    raise ValueError(f'{rid} {metric}: calculated {actual[metric]} vs frozen {expected}')
            if actual['n_conflict'] != reference['test']['conflict']['n']:
                raise ValueError(f'{rid} conflict count mismatch')
            cell = (reference['condition'], reference['architecture'])
            by_cell.setdefault(cell, []).append((reference['seed'], actual))
            calculated[rid] = actual
            print(f'verified {len(calculated)}/135 {rid}', flush=True)
    aggregate_checks = 0
    for group in aggregate['groups']:
        cell = (group['condition'], group['architecture'])
        runs = [row for _, row in sorted(by_cell[cell])]
        if len(runs) != 5:
            raise ValueError(f'Missing seed in {cell}')
        for metric in METRICS:
            expected = group['metrics'][metric]['mean']
            error = abs(statistics.mean(row[metric] for row in runs)-expected)
            max_error = max(max_error, error)
            if error > 2e-6:
                raise ValueError(f'Aggregate mismatch {cell} {metric}: {error}')
            aggregate_checks += 1
    report = {'status': 'verified', 'schema': SCHEMA, 'runs': len(calculated),
              'iid_rows': sum(r['n_iid'] for r in calculated.values()),
              'counterfactual_rows': sum(r['n_pairs']*4 for r in calculated.values()),
              'aggregate_metric_checks': aggregate_checks,
              'maximum_absolute_metric_error': max_error,
              'evidence_zip_sha256': sha(archive_path)}
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('pack')
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    v = commands.add_parser('verify')
    v.add_argument('--archive', required=True, type=Path)
    v.add_argument('--summary', required=True, type=Path)
    v.add_argument('--aggregate', required=True, type=Path)
    v.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'pack':
        pack(args.source, args.output)
    else:
        verify(args.archive, args.summary, args.aggregate, args.output)


if __name__ == '__main__':
    main()
