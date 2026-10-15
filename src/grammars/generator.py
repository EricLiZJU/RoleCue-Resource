"""Reproducible synthetic smoke -> validation gate -> pilot pipeline (stdlib only)."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil

from src.baselines.bayes_oracle import posterior_first
from src.grammars.events import derive_seed, sample_inventory, stage_inventory
from src.grammars.linearizer import QUERY, TOKENIZER_ID, realize

VARIANTS = {'reference': (1, 1), 'order_swap': (-1, -1),
            'marker_flip': (1, -1), 'both': (-1, 1)}


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as handle:
        for row in rows:
            handle.write(dumps(row) + '\n')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def make_record(event, condition, config, stage, kind='iid', variant=None, so=None):
    so = config['canonical_direction'] if so is None else so
    ro, rm = condition['r_o'], condition['r_m']
    surface_seed = derive_seed(config['seeds']['surface'], event.event_id)
    rng = random.Random(surface_seed)
    uo, um = rng.random(), rng.random()
    if variant is None:
        y = so if uo < ro else -so
        sm = y if um < rm else -y
    else:
        y, sm = VARIANTS[variant]
    tokens = realize(event, y, sm)
    candidates = [tokens[0], tokens[2]]
    train_p = posterior_first(ro, rm, so, sm)
    intervention = variant is not None
    eval_ro, eval_rm = (0.5, 0.5) if intervention else (ro, rm)
    eval_p = posterior_first(eval_ro, eval_rm, so, sm)
    return {
        'schema_version': config['schema_version'],
        'record_id': f'{stage}/{condition["id"]}/s{so}/{kind}/{event.event_id}/{variant or "sample"}',
        'event_id': event.event_id, 'event_group_id': event.group_id,
        'predicate': event.predicate,
        'true_roles': {'Agent': event.agent, 'Patient': event.patient},
        'tokens': tokens, 'candidate_order': candidates, 'marker_direction': sm,
        'surface_markers': {tokens[0]: tokens[1], tokens[2]: tokens[3]},
        'label_first_is_agent': int(y == 1), 'true_role_direction': y,
        'order_correct': y == so, 'marking_correct': sm == y, 'cue_conflict': sm != so,
        'condition_id': condition['id'],
        'training_reliability': {'r_o': ro, 'r_m': rm, 's_o': so},
        'evaluation_reliability': {'r_o': eval_ro, 'r_m': eval_rm, 's_o': so},
        'sampling_law': 'balanced_four_surface_intervention' if intervention else 'independent_bernoulli_cues',
        'oracle': {'training_p_first_agent': train_p, 'evaluation_p_first_agent': eval_p,
                   'training_supported': train_p is not None},
        'pair_id': f'{stage}/{condition["id"]}/s{so}/{kind}/{event.event_id}' if intervention else None,
        'variant': variant, 'alignment_entity': event.agent,
        'split': event.split, 'stage': stage, 'dataset_kind': kind,
        'seeds': dict(config['seeds'], event_surface=surface_seed),
        'tokenizer': TOKENIZER_ID,
        'distances': {'agent_to_predicate': 4 if y == 1 else 2,
                      'patient_to_predicate': 2 if y == 1 else 4,
                      'agent_to_query': 5 if y == 1 else 3},
    }


def generate_stage(root, master, config, stage):
    from src.grammars.validate import validate_stage
    inventory = stage_inventory(master, config['stages'][stage])
    stage_root = root / stage
    for condition in config['conditions']:
        for split, events in inventory.items():
            rows = [make_record(e, condition, config, stage) for e in events]
            random.Random(derive_seed(config['seeds']['export'], stage, split)).shuffle(rows)
            path = stage_root / condition['id'] / f'{split}.jsonl'
            write_jsonl(path, rows)
            print(f'[{stage}] {condition["id"]}/{split}: {len(rows):,} records saved', flush=True)
        path = stage_root / condition['id'] / 'counterfactual.jsonl'
        write_jsonl(path, (make_record(e, condition, config, stage, 'counterfactual', variant)
                          for e in inventory['test'] for variant in VARIANTS))
        print(f'[{stage}] {condition["id"]}/counterfactual: {4*len(inventory["test"]):,} records saved', flush=True)
    # Two reciprocal held-out events guarantee both semantic assignments in every surface cell.
    if config['boundary_conditions']:
        write_jsonl(stage_root / 'boundary.jsonl',
                    (make_record(e, c, config, stage, 'boundary', v, so)
                     for c in config['boundary_conditions'] for so in (-1, 1)
                     for e in inventory['test'][:2] for v in VARIANTS))
        write_jsonl(stage_root / 'boundary_iid.jsonl',
                    (make_record(e, c, config, stage, 'boundary_iid', so=so)
                     for c in config['boundary_conditions'] for so in (-1, 1)
                     for e in inventory['test'][:1000]))
    print(f'[{stage}] validating all saved records, splits, pairs and oracle...', flush=True)
    report = validate_stage(stage_root, inventory, config, stage)
    save_json(stage_root / 'validation.json', report)
    print(f'[{stage}] PASS: {report["total_records"]:,} records; validation gate cleared', flush=True)
    return report


def _load_excluded_groups(config):
    specification = config.get('excluded_inventory')
    if not specification:
        return set()
    path = Path(specification['path'])
    if sha256(path) != specification['sha256']:
        raise ValueError('Excluded inventory fingerprint mismatch')
    with path.open() as handle:
        return {json.loads(line)['group_id'] for line in handle}


def run(config_path, output, stages=None):
    config_path, output = Path(config_path), Path(output)
    config = json.loads(config_path.read_text())
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing dataset: {output}')
    stages = tuple(config['stages']) if stages is None else tuple(stages)
    if not stages or any(stage not in config['stages'] for stage in stages):
        raise ValueError('Unknown or empty generation stage list')
    if 'pilot' in stages and 'smoke' in config['stages'] and 'smoke' not in stages:
        raise ValueError('Pilot requires a successful smoke gate in the same run')
    master_stage = max(stages, key=lambda name: sum(config['stages'][name].values()))
    excluded_groups = _load_excluded_groups(config)
    master = sample_inventory(config['stages'][master_stage], config['seeds']['events'],
                              config['entities'], config['predicates'], excluded_groups)
    output.mkdir(parents=True)
    save_json(output / 'config.json', config)
    if excluded_groups:
        save_json(output / 'inventory/exclusions.json', {
            'source_path': config['excluded_inventory']['path'],
            'source_sha256': config['excluded_inventory']['sha256'],
            'excluded_group_count': len(excluded_groups),
            'excluded_group_ids': sorted(excluded_groups)})
    source_root = Path(__file__).resolve().parents[2]
    guide = source_root / 'docs/synthetic_data_v1.md'
    if guide.exists():
        shutil.copy2(guide, output / 'README.md')
    sources = ['src/grammars/events.py', 'src/grammars/generator.py',
               'src/grammars/linearizer.py', 'src/grammars/validate.py',
               'src/baselines/bayes_oracle.py']
    for source in sources:
        target = output / 'source' / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / source, target)
    for path in sorted((source_root / 'tests').glob('test_*.py')):
        target = output / 'source' / 'tests' / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    write_jsonl(output / 'inventory/events.jsonl', (e.to_dict() for events in master.values() for e in events))
    save_json(output / 'inventory/vocabulary.json', {
        'entities': [f'n{i:03d}' for i in range(1, config['entities']+1)],
        'predicates': [f'v{i:02d}' for i in range(1, config['predicates']+1)],
        'special_tokens': ['<NOM>', '<ACC>', QUERY],
        'possible_ordered_events': config['entities']*(config['entities']-1)*config['predicates'],
        'selected_events': sum(map(len, master.values())),
        'excluded_event_groups': len(excluded_groups),
        'role_reciprocal_groups': True})
    reports = {}
    for stage in stages:
        reports[stage] = generate_stage(output, master, config, stage)
    write_jsonl(output / 'examples.jsonl',
                (make_record(master['test'][0], c, config, stages[-1], 'counterfactual', v)
                 for c in config['conditions'] for v in VARIANTS))
    files = {str(p.relative_to(output)): {'sha256': sha256(p), 'bytes': p.stat().st_size}
             for p in sorted(output.rglob('*'))
             if p.is_file() and not any(part.startswith('._') for part in p.relative_to(output).parts)}
    # Check recorded bytes against disk before declaring the bundle complete.
    for name, info in files.items():
        if sha256(output / name) != info['sha256']:
            raise ValueError(f'Hash verification failed: {name}')
    import platform
    save_json(output / 'manifest.json', {
        'status': 'validated', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'generator_version': config['generator_version'], 'python': platform.python_version(),
        'config_sha256': sha256(output / 'config.json'), 'files': files,
        'stages': list(reports), 'record_counts': {k: v['total_records'] for k,v in reports.items()},
        'license': 'Original synthetic data: Creative Commons Attribution 4.0 International (CC BY 4.0).',
        'notes': ['Smoke is a subset of pilot; they are not independent replications.',
                  'Training seeds in pilot.yaml are model seeds, separate from these data seeds.',
                  'Counterfactual training oracle is a reference, not the intervention-distribution oracle.']})
    print(f'COMPLETE: {output} (all file SHA-256 checks passed)', flush=True)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/synthetic_v1.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    run(args.config, args.output, ('smoke',) if args.smoke_only else None)


if __name__ == '__main__':
    main()
