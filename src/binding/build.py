"""Build and validate a versioned RoleCue-Binding resource without training."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from src.binding.task import SPECIAL, SUFFIX_LENGTH, TOKENIZER, baseline_probability, oracle_from_tokens, parse, realize
from src.grammars.events import derive_seed, sample_inventory
from src.grammars.generator import save_json, sha256, write_jsonl

CONDITIONS = ((.55, .95), (.75, .75), (.95, .55))
TRAIN_CELLS = ((1, 0), (2, 0), (2, 1))
TEST_CELLS = ((1, 0), (2, 0), (2, 1))
OOD_CELLS = ((4, 2),)


def rows_for(events, ro, rm, names, verbs, split, seed):
    for event in events:
        if split == 'train':
            cells = (TRAIN_CELLS[derive_seed(seed, event.group_id) % len(TRAIN_CELLS)],)
        elif split == 'ood':
            cells = OOD_CELLS
        else:
            cells = TEST_CELLS
        for distance, depth in cells:
            for version in ('binding', 'local'):
                for slot in (0, 1):
                    yield realize(event, ro, rm, names, verbs, version, slot,
                                  distance, depth, surface_seed=seed)


def certify(rows, ro, rm):
    """Exhibit same-suffix, different-posterior, opposite-truth witnesses.

    A local suffix-only predictor must return the same number for each pair.
    Full visible strings have distinct exact posteriors, independently checked
    by enumeration in realize().
    """
    buckets = defaultdict(list)
    checked = 0
    for row in rows:
        parsed = parse(row['tokens'])
        q = oracle_from_tokens(row['tokens'], ro, rm)
        if abs(q-row['oracle_p_query_agent']) > 1e-12:
            raise AssertionError('Parser and generator oracle disagree')
        if parsed['version'] != row['version'] or parsed['query_slot'] != row['candidate_order'].index(row['query_entity']):
            raise AssertionError('Visible parser disagrees with row metadata')
        checked += 1
        if row['version'] == 'binding':
            key = (row['event_group_id'], row['distance'], row['depth'],
                   tuple(row['tokens'][-SUFFIX_LENGTH:]))
            buckets[key].append(row)
    witnesses = []
    for key, bucket in buckets.items():
        for i, a in enumerate(bucket):
            for b in bucket[i+1:]:
                if (a['label_query_is_agent'] != b['label_query_is_agent']
                        and abs(a['oracle_p_query_agent']-b['oracle_p_query_agent']) > 1e-9):
                    witnesses.append({'record_ids': [a['record_id'], b['record_id']],
                                      'same_visible_suffix': list(key[-1]),
                                      'labels': [a['label_query_is_agent'], b['label_query_is_agent']],
                                      'posteriors': [a['oracle_p_query_agent'], b['oracle_p_query_agent']]})
                    break
            else:
                continue
            break
    if not witnesses:
        raise AssertionError('No information-necessity witness found')
    methods = ('suffix', 'order_only', 'marker_only', 'unbound', 'full_oracle')
    baseline = {}
    for version in ('binding', 'local'):
        subset = [row for row in rows if row['version'] == version]
        baseline[version] = {}
        for method in methods:
            probabilities = [baseline_probability(row['tokens'], ro, rm, method) for row in subset]
            targets = [row['label_query_is_agent'] for row in subset]
            nll = -sum(y*math.log(p) + (1-y)*math.log1p(-p)
                       for y, p in zip(targets, probabilities))/len(subset)
            accuracy = sum((p >= .5) == bool(y) for p, y in zip(probabilities, targets))/len(subset)
            mae = sum(abs(p-row['oracle_p_query_agent']) for p, row in zip(probabilities, subset))/len(subset)
            baseline[version][method] = {'n': len(subset), 'nll': nll, 'accuracy': accuracy,
                                         'oracle_probability_mae': mae}
    return {'rows_validated': checked, 'same_suffix_opposite_truth_different_posterior_pairs': len(witnesses),
            'example': witnesses[0], 'baselines': baseline,
            'population_binding_suffix_posterior': .5,
            'suffix_limit': f'final {SUFFIX_LENGTH} tokens',
            'oracle_check': 'analytic posterior, independent latent-state enumeration, visible-token parser'}


def build(output, sizes=None, seed=20261015, entities=48, predicates=8):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    sizes = sizes or {'train': 1600, 'dev': 200, 'test': 200, 'ood': 200}
    names = [f'n{i:03d}' for i in range(1, entities+1)]
    verbs = [f'v{i:02d}' for i in range(1, predicates+1)]
    events = sample_inventory(sizes, seed, entities, predicates)
    output.mkdir(parents=True)
    save_json(output/'config.json', {'schema_version': TOKENIZER, 'conditions': CONDITIONS,
              'sizes_in_semantic_events': sizes, 'seed': seed, 'entities': entities,
              'predicates': predicates, 'training_cells': TRAIN_CELLS,
              'test_cells': TEST_CELLS, 'ood_cells': OOD_CELLS,
              'query_slots': [0, 1], 'versions': ['binding', 'local'],
              'local_suffix_tokens': SUFFIX_LENGTH,
              'split_unit': 'unordered entity pair plus predicate; both event orientations stay together'})
    save_json(output/'vocabulary.json', {'entities': names, 'predicates': verbs,
                                         'special_tokens': list(SPECIAL)})
    write_jsonl(output/'inventory.jsonl', (event.to_dict() for group in events.values() for event in group))
    certificates = {}
    for ro, rm in CONDITIONS:
        condition = f'ro{round(ro*100):03d}_rm{round(rm*100):03d}'
        certificates[condition] = {}
        for split in sizes:
            rows = list(rows_for(events[split], ro, rm, names, verbs, split, seed+17))
            # A separate, deterministic surface realization for OOD is not a
            # new semantic world; the split is independently held out above.
            if split in ('dev', 'test', 'ood'):
                certificates[condition][split] = certify(rows, ro, rm)
            write_jsonl(output/condition/f'{split}.jsonl', rows)
    save_json(output/'information_certificate.json', certificates)
    files = {str(path.relative_to(output)): {'sha256': sha256(path), 'bytes': path.stat().st_size}
             for path in sorted(output.rglob('*')) if path.is_file()}
    save_json(output/'manifest.json', {'status': 'validated', 'version': TOKENIZER,
              'created_utc': datetime.now(timezone.utc).isoformat(), 'files': files,
              'license': 'Not yet selected for public release',
              'notes': ['No expert linguistic review has been completed.',
                        'Artificial case markers and role mapping are stipulated.',
                        'OOD holds out event groups and increases clause count/depth.']})
    return certificates


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    sample_sizes = {'train': 80, 'dev': 20, 'test': 20, 'ood': 20} if args.smoke else None
    print(json.dumps(build(args.output, sample_sizes), indent=2))
