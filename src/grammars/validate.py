"""Fail-closed checks of saved data, semantic isolation and exact oracle."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from src.baselines.bayes_oracle import enumerate_surface, posterior_first, log_odds_posterior
from src.grammars.linearizer import parse, model_example


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    return a is b if a is None or b is None else math.isclose(a, b, abs_tol=1e-12)


def validate_record(r, config):
    parsed = parse(r['tokens'])
    for key in ('candidate_order', 'predicate', 'marker_direction'):
        require(parsed[key] == r[key], f'Surface metadata mismatch: {key}')
    first, second = r['candidate_order']
    roles = r['true_roles']
    require(set(roles.values()) == {first, second}, 'True roles do not match candidates')
    require(r['event_id'] == f'{r["predicate"]}:{roles["Agent"]}>{roles["Patient"]}', 'Event ID mismatch')
    require(r['event_group_id'] == ':'.join((r['predicate'], *sorted((first, second)))), 'Group mismatch')
    require(all(1 <= int(n[1:]) <= config['entities'] for n in (first, second)), 'Unknown entity')
    require(1 <= int(r['predicate'][1:]) <= config['predicates'], 'Unknown predicate')
    y = 1 if roles['Agent'] == first else -1
    require(r['true_role_direction'] == y and r['label_first_is_agent'] == int(y == 1), 'True role label mismatch')
    require(r['surface_markers'] == {first: r['tokens'][1], second: r['tokens'][3]}, 'Marker mapping mismatch')
    require(r['distances'] == {'agent_to_predicate': 4 if y == 1 else 2,
                              'patient_to_predicate': 2 if y == 1 else 4,
                              'agent_to_query': 5 if y == 1 else 3}, 'Distance mismatch')
    sm = r['marker_direction']
    for distribution, field in (('training_reliability', 'training_p_first_agent'),
                                ('evaluation_reliability', 'evaluation_p_first_agent')):
        params = r[distribution]
        ro, rm, so = params['r_o'], params['r_m'], params['s_o']
        formula = posterior_first(ro, rm, so, sm)
        enumeration = enumerate_surface(ro, rm, so, first, second, first if sm == 1 else second)
        require(close(formula, enumeration) and close(formula, r['oracle'][field]), 'Oracle mismatch')
        if 0 < ro < 1 and 0 < rm < 1:
            require(close(formula, log_odds_posterior(ro, rm, so, sm)), 'Log-odds mismatch')
    so = r['training_reliability']['s_o']
    require(r['order_correct'] == (y == so), 'Order correctness mismatch')
    require(r['marking_correct'] == (sm == y), 'Marking correctness mismatch')
    require(r['cue_conflict'] == (sm != so), 'Conflict mismatch')
    require(r['oracle']['training_supported'] == (r['oracle']['training_p_first_agent'] is not None), 'Support mismatch')
    require(set(model_example(r)) == {'tokens', 'label'}, 'Model input allowlist failed')


def validate_pairs(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r['pair_id']].append(r)
    for pair, records in groups.items():
        require(pair is not None and len(records) == 4, 'Incomplete counterfactual group')
        variants = {r['variant']: r for r in records}
        require(set(variants) == {'reference', 'order_swap', 'marker_flip', 'both'}, 'Missing variant')
        ref = variants['reference']
        for r in records:
            require(r['event_id'] == ref['event_id'] and r['true_roles'] == ref['true_roles'], 'Pair changes semantics')
            require(Counter(r['tokens']) == Counter(ref['tokens']), 'Pair changes length or token counts')
            require(r['oracle']['evaluation_p_first_agent'] == 0.5, 'Wrong intervention oracle')
            require(r['evaluation_reliability']['r_o'] == r['evaluation_reliability']['r_m'] == 0.5,
                    'Wrong intervention distribution')
        swap = variants['order_swap']
        flip = variants['marker_flip']
        both = variants['both']
        require(swap['candidate_order'] == ref['candidate_order'][::-1]
                and swap['surface_markers'] == ref['surface_markers'], 'Order swap does not preserve entity markers')
        require(flip['candidate_order'] == ref['candidate_order']
                and flip['surface_markers'] != ref['surface_markers'], 'Marker flip invalid')
        require(both['candidate_order'] == ref['candidate_order'][::-1]
                and both['surface_markers'] == flip['surface_markers'], 'Combined intervention invalid')
    return len(groups)


def validate_inventory(inventory):
    all_groups = set()
    balance = {}
    for split, events in inventory.items():
        groups = {e.group_id for e in events}
        require(not all_groups & groups, 'Event group leakage across splits')
        all_groups.update(groups)
        require(len({e.event_id for e in events}) == len(events), 'Duplicate event')
        require(len(groups)*2 == len(events), 'Missing reciprocal events')
        agents = Counter(e.agent for e in events)
        patients = Counter(e.patient for e in events)
        require(agents == patients, 'Entity roles are not balanced')
        verbs = Counter(e.predicate for e in events)
        require(max(verbs.values())-min(verbs.values()) <= 2, 'Predicate frequency imbalance')
        balance[split] = {'events': len(events), 'groups': len(groups), 'entities_observed': len(agents),
                          'entity_agent_count_min': min(agents.values()), 'entity_agent_count_max': max(agents.values()),
                          'entity_roles_exactly_balanced': True, 'predicate_counts': dict(sorted(verbs.items()))}
    return balance


def statistics(rows, ro, rm):
    n = len(rows)
    expected = {'order_correct': ro, 'marking_correct': rm,
                'cue_conflict': ro*(1-rm)+(1-ro)*rm,
                'both_correct': ro*rm}
    rates = {k: sum((r['order_correct'] and r['marking_correct']) if k == 'both_correct' else r[k]
                    for r in rows)/n for k in expected}
    # Hoeffding bound for each predeclared Bernoulli statistic; delta=1e-6 per check.
    tolerance = math.sqrt(math.log(2/1e-6)/(2*n))
    for key, value in expected.items():
        require(abs(rates[key]-value) <= tolerance, f'Empirical reliability outside bound: {key}')
    losses, scores = [], []
    for r in rows:
        p = r['oracle']['evaluation_p_first_agent']
        require(p is not None, 'IID observation has zero support')
        target = r['label_first_is_agent']
        prob = p if target else 1-p
        require(prob > 0, 'IID oracle assigned zero probability to truth')
        losses.append(-math.log(prob))
        scores.append(0.5 if close(p, 0.5) else float((p > 0.5) == bool(target)))
    return {'records': n, 'expected': expected, 'empirical': rates,
            'hoeffding_tolerance': tolerance, 'delta_per_statistic': 1e-6,
            'oracle_accuracy_empirical_ties_half': sum(scores)/n,
            'oracle_accuracy_population': max(ro, rm),
            'oracle_nll_empirical': sum(losses)/n}


def validate_stage(root, inventory, config, stage):
    balance = validate_inventory(inventory)
    expected = {split: {e.event_id for e in events} for split, events in inventory.items()}
    seen_ids, stats, total, pair_count = set(), {}, 0, 0
    for path in sorted(Path(root).rglob('*.jsonl')):
        if any(part.startswith('._') for part in path.relative_to(root).parts):
            continue  # macOS AppleDouble metadata on ExFAT is not a data record.
        with path.open() as handle:
            rows = [json.loads(line) for line in handle]
        require(bool(rows), f'Empty dataset: {path}')
        for r in rows:
            validate_record(r, config)
            require(r['record_id'] not in seen_ids, 'Duplicate record ID')
            seen_ids.add(r['record_id'])
            require(r['event_id'] in expected[r['split']], 'Record belongs to wrong split')
            require(r['stage'] == stage, 'Wrong stage')
            is_boundary = path.stem.startswith('boundary')
            settings = config['boundary_conditions'] if is_boundary else config['conditions']
            matches = [c for c in settings if c['id'] == r['condition_id']]
            require(len(matches) == 1, 'Unknown reliability condition')
            params = r['training_reliability']
            require(params['r_o'] == matches[0]['r_o'] and params['r_m'] == matches[0]['r_m'],
                    'Reliability differs from frozen config')
            require(params['s_o'] in (-1, 1) if is_boundary else params['s_o'] == config['canonical_direction'],
                    'Canonical direction differs from config')
            from src.grammars.events import derive_seed
            require(r['seeds'] == dict(config['seeds'], event_surface=derive_seed(config['seeds']['surface'], r['event_id'])),
                    'Seed metadata differs from frozen config')
            if path.stem in ('train', 'dev', 'test'):
                require(r['split'] == path.stem and r['dataset_kind'] == 'iid' and r['pair_id'] is None,
                        'IID dataset metadata mismatch')
                require(r['evaluation_reliability'] == params, 'IID evaluation distribution mismatch')
            else:
                require(r['split'] == 'test' and r['dataset_kind'] == path.stem, 'Diagnostic split mismatch')
        total += len(rows)
        if path.stem in ('counterfactual', 'boundary'):
            pair_count += validate_pairs(rows)
        if path.stem in ('train', 'dev', 'test'):
            condition = next(c for c in config['conditions'] if c['id'] == path.parent.name)
            require(len(rows) == len(expected[path.stem]), 'Wrong split count')
            require({r['event_id'] for r in rows} == expected[path.stem], 'Wrong split coverage')
            stats[str(path.relative_to(root))] = statistics(rows, condition['r_o'], condition['r_m'])
        if path.stem == 'counterfactual':
            require(len(rows) == 4*len(expected['test']), 'Wrong counterfactual count')
            require({r['event_id'] for r in rows} == expected['test'], 'Wrong counterfactual coverage')
        if path.stem == 'boundary_iid':
            groups = defaultdict(list)
            for r in rows:
                groups[(r['condition_id'], r['training_reliability']['s_o'])].append(r)
            for (condition, so), group in groups.items():
                params = group[0]['training_reliability']
                stats[f'boundary_iid/{condition}/s{so}'] = statistics(group, params['r_o'], params['r_m'])
    expected_total = len(config['conditions'])*(sum(map(len, inventory.values()))+4*len(inventory['test']))
    expected_total += len(config['boundary_conditions'])*2*(8+min(1000, len(inventory['test'])))
    require(total == expected_total, 'Unexpected total record count or missing file')
    return {'status': 'passed', 'stage': stage, 'total_records': total, 'pair_groups': pair_count,
            'inventory': balance, 'statistics': stats,
            'checks': ['surface parse and truth consistency', 'oracle formula vs latent enumeration',
                       'log-odds agreement in interior', 'group-level split isolation',
                       'exact within-entity role balance', 'four-cell entity-aligned interventions',
                       'per-file counts and global IDs', 'empirical cue/conflict/joint probabilities',
                       'tokens-only model input allowlist'],
            'limitations': ['Marginal entity frequencies are sampled, not exactly equal.',
                            'No neural model has been trained by this validation.']}


def validate_bundle(root):
    from src.grammars.events import sample_inventory, stage_inventory
    from src.grammars.generator import sha256
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    require(manifest['status'] == 'validated', 'Incomplete bundle')
    for name, info in manifest['files'].items():
        path = root / name
        require(path.is_file() and sha256(path) == info['sha256'] and path.stat().st_size == info['bytes'],
                f'File integrity failure: {name}')
    config = json.loads((root / 'config.json').read_text())
    master_stage = max(manifest['stages'], key=lambda name: sum(config['stages'][name].values()))
    excluded = set()
    specification = config.get('excluded_inventory')
    if specification:
        saved_exclusions = root/'inventory/exclusions.json'
        require(saved_exclusions.is_file(), 'Missing bundled exclusion proof')
        proof = json.loads(saved_exclusions.read_text())
        require(proof['source_sha256'] == specification['sha256'], 'Exclusion proof fingerprint mismatch')
        excluded = set(proof['excluded_group_ids'])
        require(len(excluded) == proof['excluded_group_count'], 'Exclusion proof count mismatch')
    master = sample_inventory(config['stages'][master_stage], config['seeds']['events'],
                              config['entities'], config['predicates'], excluded)
    with (root / 'inventory/events.jsonl').open() as handle:
        saved = [json.loads(line) for line in handle]
    require(saved == [e.to_dict() for events in master.values() for e in events], 'Master inventory mismatch')
    reports = {}
    for stage in manifest['stages']:
        report = validate_stage(root / stage, stage_inventory(master, config['stages'][stage]), config, stage)
        require(report == json.loads((root / stage / 'validation.json').read_text()), 'Stored validation report mismatch')
        reports[stage] = report['total_records']
        print(f'[{stage}] independently revalidated {report["total_records"]:,} records', flush=True)
    return reports


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle')
    print(validate_bundle(parser.parse_args().bundle))
