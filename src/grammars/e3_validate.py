"""Independent, streaming validation for a saved E3 bundle."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from src.grammars.e3_generator import validate_e3_rows
from src.grammars.generator import save_json, sha256


def validate_bundle(root):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['status'] != 'validated':
        raise ValueError('E3 bundle is not validated')
    for name, info in manifest['files'].items():
        path = root/name
        if not path.is_file() or path.stat().st_size != info['bytes'] or sha256(path) != info['sha256']:
            raise ValueError(f'E3 bundle integrity failure: {name}')
    config = json.loads((root/'config.json').read_text())
    stage = manifest['stage']
    inventory = {split: set() for split in config['stages'][stage]}
    groups = {split: set() for split in inventory}
    with (root/'inventory/events.jsonl').open() as handle:
        for line in handle:
            row = json.loads(line); split = row['split']
            if row['event_id'] in inventory[split]:
                raise ValueError('Duplicate E3 inventory event')
            inventory[split].add(row['event_id']); groups[split].add(row['group_id'])
    if any(groups[a] & groups[b] for i, a in enumerate(groups) for b in list(groups)[i+1:]):
        raise ValueError('E3 inventory semantic leakage')
    for split, expected in config['stages'][stage].items():
        if len(inventory[split]) != expected or len(groups[split])*2 != expected:
            raise ValueError('E3 inventory count/reciprocal closure mismatch')
    if config.get('excluded_inventory'):
        proof = json.loads((root/'inventory/exclusions.json').read_text())
        if proof['source_sha256'] != config['excluded_inventory']['sha256']:
            raise ValueError('E3 exclusion proof mismatch')
        excluded = set(proof['excluded_group_ids'])
        if any(excluded & values for values in groups.values()):
            raise ValueError('Engineering-smoke group leaked into formal E3')
    files, total, global_ids, max_length = {}, 0, set(), 0
    for condition in config['conditions']:
        cid = condition['id']
        for split in ('train', 'dev', 'test'):
            path = root/stage/cid/f'{split}.jsonl'
            count = 0; events = set(); cells = Counter(); cue = Counter()
            with path.open() as handle:
                for line in handle:
                    row = json.loads(line); validate_e3_rows([row], config, split)
                    if row['record_id'] in global_ids:
                        raise ValueError('Duplicate E3 global record ID')
                    global_ids.add(row['record_id']); events.add(row['event_id'])
                    cells[row['structure_cell']] += 1
                    cue['order_correct'] += row['order_correct']; cue['marking_correct'] += row['marking_correct']
                    max_length = max(max_length, len(row['tokens'])); count += 1
            if count != config['stages'][stage][split] or events != inventory[split]:
                raise ValueError('E3 file coverage mismatch')
            tolerance = math.sqrt(math.log(2/1e-6)/(2*count))
            if (abs(cue['order_correct']/count-condition['r_o']) > tolerance
                    or abs(cue['marking_correct']/count-condition['r_m']) > tolerance):
                raise ValueError('E3 empirical reliability outside frozen tolerance')
            files[str(path.relative_to(root))] = {'records': count, 'cells': dict(cells)}
            total += count
        path = root/stage/cid/'counterfactual.jsonl'
        count = groups_checked = 0; buffer = []
        with path.open() as handle:
            for line in handle:
                row = json.loads(line); buffer.append(row); count += 1
                max_length = max(max_length, len(row['tokens']))
                if len(buffer) == 4:
                    validate_e3_rows(buffer, config, 'test', True)
                    if len({r['pair_id'] for r in buffer}) != 1:
                        raise ValueError('Non-contiguous E3 pair group')
                    groups_checked += 1; buffer = []
        if buffer:
            raise ValueError('Incomplete final E3 diagnostic group')
        expected = config['diagnostic_events']*len(config['diagnostic_structure_cells'])*4
        if count != expected:
            raise ValueError('E3 diagnostic count mismatch')
        files[str(path.relative_to(root))] = {'records': count, 'pair_groups': groups_checked}
        total += count
    if max_length > config['max_length']:
        raise ValueError('E3 saved sequence exceeds frozen no-truncation context')
    return {'status': 'passed', 'bundle_manifest_sha256': sha256(root/'manifest.json'),
            'manifest_files_verified': len(manifest['files']), 'records_verified': total,
            'unique_iid_record_ids': len(global_ids), 'max_sequence_length': max_length,
            'semantic_groups': {key: len(value) for key, value in groups.items()}, 'files': files,
            'checks': ['all manifest hashes and byte sizes', 'all saved records parsed',
                       'formula, enumeration and log-odds oracle agreement',
                       'train/dev/test semantic group isolation and reciprocal closure',
                       'engineering-smoke exclusion', 'counterfactual four-cell closure',
                       'frozen structure cells and context limit', 'no truncation']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root'); parser.add_argument('--output', required=True)
    args = parser.parse_args(); result = validate_bundle(args.root); save_json(Path(args.output), result)
    print(json.dumps({k:v for k,v in result.items() if k != 'files'}, indent=2))
