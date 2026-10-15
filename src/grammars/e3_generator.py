"""Generate E3 distance/hierarchy data with explicit structure and unchanged cue oracle."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil

from src.baselines.bayes_oracle import enumerate_surface, log_odds_posterior, posterior_first
from src.grammars.events import derive_seed, sample_inventory
from src.grammars.generator import VARIANTS, save_json, sha256, write_jsonl
from src.grammars.linearizer import (E3_SPECIAL_TOKENS, E3_TOKENIZER_ID, QUERY,
                                     parse_structured)


def _distractors(event, count, entities, seed):
    pool = [f'n{i:03d}' for i in range(1, entities+1)
            if f'n{i:03d}' not in (event.agent, event.patient)]
    rng = random.Random(seed)
    rng.shuffle(pool)
    if count > len(pool):
        raise ValueError('Not enough distinct distractor entities')
    return pool[:count]


def _structured_material(event, insertion_tokens, depth, config, seed):
    """Create exactly insertion_tokens meaningful tokens with balanced nesting."""
    if insertion_tokens < 0 or depth < 0:
        raise ValueError('Negative E3 structure setting')
    if insertion_tokens == 0:
        if depth != 0:
            raise ValueError('Positive depth requires insertion material')
        return [], {'type': 'empty_modifier', 'depth': 0, 'children': []}, []
    minimum = 5*depth
    if insertion_tokens < minimum:
        raise ValueError('Insertion span too short for requested depth')
    distractor_count = max(depth, (insertion_tokens+2)//3)
    nouns = _distractors(event, distractor_count, config['entities'], seed)
    predicate_number = (int(event.predicate[1:]) % config['predicates']) + 1
    subverb = f'v{predicate_number:02d}'
    tokens, used = [], []
    # Each nested clause is structurally complete and contains a competing noun
    # with an overt case marker and predicate.
    for level in range(depth):
        noun = nouns[len(used)]; used.append(noun)
        tokens += ['<REL>', noun, '<NOM>' if level % 2 == 0 else '<ACC>', subverb]
    remaining = insertion_tokens - len(tokens) - depth
    cursor = len(used)
    while remaining >= 3:
        noun = nouns[cursor]; cursor += 1; used.append(noun)
        tokens += ['<MOD>', noun, '<COMP>']
        remaining -= 3
    tokens += ['<MOD>'] * remaining
    tokens += ['</REL>'] * depth
    if len(tokens) != insertion_tokens:
        raise AssertionError('E3 insertion length construction failed')
    tree = {'type': 'modifier', 'depth': depth,
            'children': [{'type': 'relative_clause', 'level': i+1,
                          'competitor': noun, 'predicate': subverb}
                         for i, noun in enumerate(used[:depth])],
            'flat_modifiers': used[depth:]}
    return tokens, tree, used


def make_e3_record(event, condition, config, stage, structure, kind='iid', variant=None):
    ro, rm, so = condition['r_o'], condition['r_m'], config['canonical_direction']
    surface_seed = derive_seed(config['seeds']['surface'], event.event_id)
    rng = random.Random(surface_seed)
    if variant is None:
        y = so if rng.random() < ro else -so
        sm = y if rng.random() < rm else -y
    else:
        y, sm = VARIANTS[variant]
    first, second = ((event.agent, event.patient) if y == 1
                     else (event.patient, event.agent))
    marks = ('<NOM>', '<ACC>') if sm == 1 else ('<ACC>', '<NOM>')
    insertion = structure['insertion_tokens']
    depth = structure['depth']
    material, modifier_tree, competitors = _structured_material(
        event, insertion, depth, config,
        derive_seed(config['seeds']['structure'], event.event_id, insertion, depth))
    tokens = ['<MAIN>', first, marks[0], *material, second, marks[1], event.predicate,
              '</MAIN>', QUERY]
    parse_structured(tokens, [first, second])
    first_i, second_i, predicate_i, query_i = (tokens.index(first), tokens.index(second),
                                                tokens.index(event.predicate), len(tokens)-1)
    train_p = posterior_first(ro, rm, so, sm)
    intervention = variant is not None
    eval_p = posterior_first(0.5 if intervention else ro, 0.5 if intervention else rm, so, sm)
    cell = f'd{insertion}_h{depth}'
    pair_id = (f'{stage}/{condition["id"]}/{kind}/{event.event_id}/{cell}'
               if intervention else None)
    return {
        'schema_version': config['schema_version'],
        'record_id': f'{stage}/{condition["id"]}/{kind}/{event.event_id}/{cell}/{variant or "sample"}',
        'event_id': event.event_id, 'event_group_id': event.group_id,
        'predicate': event.predicate,
        'true_roles': {'Agent': event.agent, 'Patient': event.patient},
        'tokens': tokens, 'candidate_order': [first, second], 'marker_direction': sm,
        'surface_markers': {first: marks[0], second: marks[1]},
        'label_first_is_agent': int(y == 1), 'true_role_direction': y,
        'order_correct': y == so, 'marking_correct': sm == y, 'cue_conflict': sm != so,
        'condition_id': condition['id'],
        'training_reliability': {'r_o': ro, 'r_m': rm, 's_o': so},
        'evaluation_reliability': {'r_o': 0.5 if intervention else ro,
                                   'r_m': 0.5 if intervention else rm, 's_o': so},
        'sampling_law': ('balanced_four_surface_intervention' if intervention
                         else 'independent_bernoulli_cues'),
        'oracle': {'training_p_first_agent': train_p,
                   'evaluation_p_first_agent': eval_p, 'training_supported': train_p is not None},
        'pair_id': pair_id, 'variant': variant, 'alignment_entity': event.agent,
        'split': event.split, 'stage': stage, 'dataset_kind': kind,
        'seeds': dict(config['seeds'], event_surface=surface_seed),
        'tokenizer': E3_TOKENIZER_ID,
        'structure_cell': cell, 'syntax_tree': {'type': 'main_clause',
            'arguments': [first, second], 'modifier': modifier_tree, 'predicate': event.predicate},
        'competitor_entities': competitors,
        'distances': {
            'first_np_to_predicate': predicate_i-first_i,
            'second_np_to_predicate': predicate_i-second_i,
            'first_np_to_query': query_i-first_i,
            'target_case_to_nearest_competitor': (min(abs(tokens.index(noun)-tokens.index(marks[0]))
                                                       for noun in competitors)
                                                  if competitors else None),
            'linear_insertion_tokens': insertion, 'tree_depth': depth,
            'sequence_length': len(tokens)},
    }


def _iid_structure(event, config, split):
    cells = config['train_structure_cells'] if split in ('train', 'dev', 'test') else []
    return cells[derive_seed(config['seeds']['structure'], event.event_id, split) % len(cells)]


def validate_e3_rows(rows, config, split, diagnostic=False):
    if not rows:
        raise ValueError('Empty E3 data')
    allowed = {(c['insertion_tokens'], c['depth']) for c in
               (config['diagnostic_structure_cells'] if diagnostic else config['train_structure_cells'])}
    ids = set()
    for row in rows:
        if row['record_id'] in ids:
            raise ValueError('Duplicate E3 record id')
        ids.add(row['record_id'])
        if row['split'] != split or row['tokenizer'] != E3_TOKENIZER_ID:
            raise ValueError('E3 split/tokenizer mismatch')
        parsed = parse_structured(row['tokens'], row['candidate_order'])
        if parsed['length'] > config['max_length']:
            raise ValueError('E3 context overflow; truncation is forbidden')
        d = row['distances']
        if (d['linear_insertion_tokens'], d['tree_depth']) not in allowed:
            raise ValueError('Unfrozen E3 structure cell')
        if d['sequence_length'] != len(row['tokens']):
            raise ValueError('E3 sequence length metadata mismatch')
        expected = posterior_first(row['evaluation_reliability']['r_o'],
                                   row['evaluation_reliability']['r_m'],
                                   row['evaluation_reliability']['s_o'],
                                   row['marker_direction'])
        first, second = row['candidate_order']
        enumerated = enumerate_surface(row['evaluation_reliability']['r_o'],
                                       row['evaluation_reliability']['r_m'],
                                       row['evaluation_reliability']['s_o'], first, second,
                                       first if row['marker_direction'] == 1 else second)
        if (abs(expected-row['oracle']['evaluation_p_first_agent']) > 1e-12
                or abs(expected-enumerated) > 1e-12):
            raise ValueError('E3 oracle mismatch')
        params = row['training_reliability']
        training_expected = posterior_first(params['r_o'], params['r_m'], params['s_o'],
                                            row['marker_direction'])
        training_enumerated = enumerate_surface(params['r_o'], params['r_m'], params['s_o'],
                                                first, second,
                                                first if row['marker_direction'] == 1 else second)
        if (abs(training_expected-row['oracle']['training_p_first_agent']) > 1e-12
                or abs(training_expected-training_enumerated) > 1e-12
                or abs(training_expected-log_odds_posterior(
                    params['r_o'], params['r_m'], params['s_o'], row['marker_direction'])) > 1e-12):
            raise ValueError('E3 training oracle mismatch')
    if diagnostic:
        groups = defaultdict(dict)
        for row in rows:
            groups[row['pair_id']][row['variant']] = row
        variants = {'reference', 'order_swap', 'marker_flip', 'both'}
        if any(set(group) != variants for group in groups.values()):
            raise ValueError('Incomplete E3 counterfactual group')
        for group in groups.values():
            ref, swap, flip, both = (group[name] for name in
                                     ('reference', 'order_swap', 'marker_flip', 'both'))
            if any(row['true_roles'] != ref['true_roles'] for row in group.values()):
                raise ValueError('E3 counterfactual changes semantic truth')
            if any(Counter(row['tokens']) != Counter(ref['tokens']) for row in group.values()):
                raise ValueError('E3 counterfactual changes lexical multiset or length')
            if (swap['candidate_order'] != ref['candidate_order'][::-1]
                    or flip['candidate_order'] != ref['candidate_order']
                    or both['candidate_order'] != ref['candidate_order'][::-1]):
                raise ValueError('E3 counterfactual candidate alignment failure')
    return {'records': len(rows), 'max_length': max(len(r['tokens']) for r in rows),
            'min_length': min(len(r['tokens']) for r in rows),
            'structure_cells': sorted({r['structure_cell'] for r in rows})}


def run(config_path, output, stage):
    config_path, output = Path(config_path), Path(output)
    config = json.loads(config_path.read_text())
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite E3 dataset: {output}')
    if stage not in config['stages']:
        raise ValueError('Unknown E3 stage')
    sizes = config['stages'][stage]
    excluded = set()
    specification = config.get('excluded_inventory')
    if specification:
        exclusion_path = Path(specification['path'])
        if sha256(exclusion_path) != specification['sha256']:
            raise ValueError('E3 excluded inventory fingerprint mismatch')
        with exclusion_path.open() as handle:
            excluded = {json.loads(line)['group_id'] for line in handle}
    inventory = sample_inventory(sizes, config['seeds']['events'], config['entities'],
                                 config['predicates'], excluded)
    output.mkdir(parents=True)
    save_json(output/'config.json', config)
    write_jsonl(output/'inventory/events.jsonl',
                (event.to_dict() for values in inventory.values() for event in values))
    if excluded:
        save_json(output/'inventory/exclusions.json', {
            'source_path': specification['path'], 'source_sha256': specification['sha256'],
            'excluded_group_count': len(excluded), 'excluded_group_ids': sorted(excluded)})
    save_json(output/'inventory/vocabulary.json', {
        'entities': [f'n{i:03d}' for i in range(1, config['entities']+1)],
        'predicates': [f'v{i:02d}' for i in range(1, config['predicates']+1)],
        'special_tokens': ['<NOM>', '<ACC>', QUERY, *E3_SPECIAL_TOKENS],
        'tokenizer': E3_TOKENIZER_ID})
    reports = {}
    for condition in config['conditions']:
        reports[condition['id']] = {}
        for split, events in inventory.items():
            rows = [make_e3_record(event, condition, config, stage,
                                   _iid_structure(event, config, split)) for event in events]
            random.Random(derive_seed(config['seeds']['export'], condition['id'], split)).shuffle(rows)
            validate_e3_rows(rows, config, split)
            write_jsonl(output/stage/condition['id']/f'{split}.jsonl', rows)
            reports[condition['id']][split] = {'records': len(rows)}
        diagnostic_events = inventory['test'][:config['diagnostic_events']]
        diagnostic = [make_e3_record(event, condition, config, stage, cell,
                                     'counterfactual', variant)
                      for event in diagnostic_events
                      for cell in config['diagnostic_structure_cells']
                      for variant in VARIANTS]
        validate_e3_rows(diagnostic, config, 'test', True)
        write_jsonl(output/stage/condition['id']/'counterfactual.jsonl', diagnostic)
        reports[condition['id']]['counterfactual'] = {'records': len(diagnostic)}
    # Independent semantic split proof.
    groups = {split: {event.group_id for event in events} for split, events in inventory.items()}
    if any(groups[a] & groups[b] for a in groups for b in groups if a < b):
        raise ValueError('E3 event group leakage')
    save_json(output/stage/'validation.json', {'status': 'passed', 'reports': reports,
        'checks': ['semantic group isolation', 'surface parser', 'unchanged two-cue oracle',
                   'explicit syntax tree', 'distance recomputation', 'no context overflow']})
    source_root = Path(__file__).resolve().parents[2]
    for source in ['src/grammars/e3_generator.py', 'src/grammars/linearizer.py',
                   'src/baselines/bayes_oracle.py']:
        target = output/'source'/source; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root/source, target)
    files = {str(path.relative_to(output)): {'sha256': sha256(path), 'bytes': path.stat().st_size}
             for path in sorted(output.rglob('*')) if path.is_file()}
    save_json(output/'manifest.json', {'status': 'validated',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'stage': stage,
        'config_sha256': sha256(output/'config.json'), 'files': files,
        'license': 'Project LICENSE remains to be finalized; no new distribution license asserted.'})
    print(f'COMPLETE E3 {stage}: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--stage', choices=('smoke', 'formal'), default='smoke')
    args = parser.parse_args()
    run(args.config, args.output, args.stage)
