"""Synthetic transitive clauses with independent cue noise and irrelevant clauses.

This is a stipulated artificial construction, not a claim about a natural language.
The only main-clause arguments are the two names inside MAIN. REL clauses have
their own arguments and case markers; their roles never determine the main answer.
"""
import math
import random
import re

from src.baselines.bayes_oracle import posterior_first, enumerate_surface
from src.grammars.events import derive_seed

TOKENIZER = 'rolecue_binding_v1'
SPECIAL = ('<PAD>', '<MAIN>', '</MAIN>', '<REL>', '</REL>', '<NOM>', '<ACC>',
           '<BIND>', '<LOCAL>', '<MASK>', '<QUERY>', '<Q_FIRST>', '<Q_SECOND>',
           '<NOM_FIRST>', '<NOM_SECOND>')
SUFFIX_LENGTH = 9


def _clause(subject, obj, verb):
    return ['<REL>', subject, '<NOM>', obj, '<ACC>', verb, '</REL>']


def _distractors(event, names, verbs, count, depth):
    """Depth 0 is sequential; positive depth nests one REL at a time.

    The nested and sequential strings have exactly the same tokens and length.
    Entities in distractors are disjoint from both main arguments.
    """
    eligible = [name for name in names if name not in (event.agent, event.patient)]
    rng = random.Random(derive_seed(12371, event.group_id, 'distractors'))
    rng.shuffle(eligible)
    if len(eligible) < count * 2:
        raise ValueError('Not enough distinct distractor entities')
    clauses = [_clause(eligible[2*i], eligible[2*i+1], verbs[i % len(verbs)])
               for i in range(count)]
    if depth > count-1 or depth < 0:
        raise ValueError('Invalid nesting depth')
    # The first (count-depth-1) clauses precede a nested chain. The final
    # clause remains the same nearest-QUERY suffix across all structures.
    plain = clauses[:count-depth-1]
    chain = clauses[count-depth-1:]
    output = [token for clause in plain for token in clause]
    if len(chain) == 1:
        output.extend(chain[0])
    else:
        for clause in chain[:-1]:
            output.extend(clause[:-1])
        output.extend(chain[-1])
        output.extend(clause[-1] for clause in reversed(chain[:-1]))
    return output


def realize(event, ro, rm, names, verbs, version, query_slot, distance=1, depth=0,
            surface_seed=41):
    """Generate one row. Query slot and task version are crossed, not sampled.

    The latent event and cue noises are sampled independently of the query.
    Both queried names are emitted for every realized surface, preserving balance.
    """
    if version not in ('binding', 'local') or query_slot not in (0, 1):
        raise ValueError('Unknown task version or query slot')
    if not (0 < ro < 1 and 0 < rm < 1):
        raise ValueError('Reliabilities must be interior probabilities')
    rng = random.Random(derive_seed(surface_seed, event.event_id))
    y = 1 if rng.random() < ro else -1
    sm = y if rng.random() < rm else -y
    first, second = (event.agent, event.patient) if y == 1 else (event.patient, event.agent)
    mark1, mark2 = ('<NOM>', '<ACC>') if sm == 1 else ('<ACC>', '<NOM>')
    query = (first, second)[query_slot]
    main = ['<MAIN>', first, mark1, second, mark2, event.predicate, '</MAIN>']
    filler = _distractors(event, names, verbs, distance, depth)
    if version == 'local':
        suffix = ['<LOCAL>', '<Q_FIRST>' if query_slot == 0 else '<Q_SECOND>',
                  '<NOM_FIRST>' if sm == 1 else '<NOM_SECOND>', '<QUERY>', query]
    else:
        suffix = ['<BIND>', '<MASK>', '<MASK>', '<QUERY>', query]
    tokens = main + filler + suffix
    q_first = posterior_first(ro, rm, 1, sm)
    q_query = q_first if query_slot == 0 else 1-q_first
    independent = enumerate_surface(ro, rm, 1, first, second, first if sm == 1 else second)
    if independent is None or not math.isclose(q_first, independent, abs_tol=1e-12):
        raise AssertionError('Independent enumeration disagrees with analytic posterior')
    return {'schema_version': TOKENIZER, 'tokenizer': TOKENIZER,
            'record_id': f'{event.event_id}:{version}:q{query_slot}:d{distance}:h{depth}',
            'event_id': event.event_id, 'event_group_id': event.group_id, 'split': event.split,
            'condition': f'ro{round(ro*100):03d}_rm{round(rm*100):03d}',
            'version': version, 'distance': distance, 'depth': depth,
            'tokens': tokens, 'candidate_order': [first, second], 'query_entity': query,
            'label_query_is_agent': int(query == event.agent),
            'cue_conflict': sm != 1, 'oracle_p_query_agent': q_query,
            'oracle_p_first_agent': q_first, 'r_o': ro, 'r_m': rm,
            'surface_seed': surface_seed}


def parse(tokens):
    """Parse only visible tokens, returning sufficient statistics and the query.

    The role parser never reads generation metadata or the gold target.
    """
    if len(tokens) < 19 or tokens[0] != '<MAIN>' or tokens[6] != '</MAIN>':
        raise ValueError('Malformed main clause')
    first, mark1, second, mark2, verb = tokens[1:6]
    if (first == second or not all(re.fullmatch(r'n\d{3}', x) for x in (first, second))
            or not re.fullmatch(r'v\d{2}', verb)
            or (mark1, mark2) not in (('<NOM>', '<ACC>'), ('<ACC>', '<NOM>'))):
        raise ValueError('Malformed main arguments or case markers')
    if tokens[-2] != '<QUERY>' or tokens[-1] not in (first, second):
        raise ValueError('Query must name exactly one main argument')
    if tokens[-5] not in ('<BIND>', '<LOCAL>'):
        raise ValueError('Unknown task mode')
    filler = tokens[7:-5]
    used = {first, second}
    def clause(i):
        if filler[i:i+1] != ['<REL>'] or i+6 >= len(filler):
            raise ValueError('Malformed relative clause opening')
        subject, nominative, obj, accusative, predicate = filler[i+1:i+6]
        if (not re.fullmatch(r'n\d{3}', subject) or not re.fullmatch(r'n\d{3}', obj)
                or subject in used or obj in used or subject == obj
                or nominative != '<NOM>' or accusative != '<ACC>'
                or not re.fullmatch(r'v\d{2}', predicate)):
            raise ValueError('Illegal distractor arguments or role markers')
        used.update((subject, obj))
        i += 6
        while i < len(filler) and filler[i] == '<REL>':
            i = clause(i)
        if i >= len(filler) or filler[i] != '</REL>':
            raise ValueError('Unclosed relative clause')
        return i+1
    i = 0
    count = 0
    while i < len(filler):
        i = clause(i)
        count += 1
    if count == 0:
        raise ValueError('Missing distractor')
    if tokens[-5] == '<LOCAL>':
        expected = ('<Q_FIRST>' if tokens[-1] == first else '<Q_SECOND>',
                    '<NOM_FIRST>' if mark1 == '<NOM>' else '<NOM_SECOND>')
        if tuple(tokens[-4:-2]) != expected:
            raise ValueError('Local summary disagrees with main clause')
    elif tokens[-4:-2] != ['<MASK>', '<MASK>']:
        raise ValueError('Binding mode leaks a local summary')
    return {'candidate_order': (first, second), 'marker_direction': 1 if mark1 == '<NOM>' else -1,
            'query_slot': 0 if tokens[-1] == first else 1, 'version': 'local' if tokens[-5] == '<LOCAL>' else 'binding',
            'predicate': verb, 'distractor_clauses': filler.count('<REL>')}


def oracle_from_tokens(tokens, ro, rm):
    parsed = parse(tokens)
    q_first = posterior_first(ro, rm, 1, parsed['marker_direction'])
    return q_first if parsed['query_slot'] == 0 else 1-q_first


def oracle_from_local_suffix(tokens, ro, rm):
    """An observer restricted to the final five tokens of the local condition."""
    suffix = tokens[-5:]
    if (len(suffix) != 5 or suffix[0] != '<LOCAL>' or suffix[3] != '<QUERY>'
            or suffix[1] not in ('<Q_FIRST>', '<Q_SECOND>')
            or suffix[2] not in ('<NOM_FIRST>', '<NOM_SECOND>')):
        raise ValueError('No sufficient local summary')
    q_first = posterior_first(ro, rm, 1, 1 if suffix[2] == '<NOM_FIRST>' else -1)
    return q_first if suffix[1] == '<Q_FIRST>' else 1-q_first


def baseline_probability(tokens, ro, rm, method):
    parsed = parse(tokens)
    if method == 'full_oracle':
        return oracle_from_tokens(tokens, ro, rm)
    if method == 'order_only':
        return ro if parsed['query_slot'] == 0 else 1-ro
    if method == 'marker_only':
        nom_first = parsed['marker_direction'] == 1
        query_nom = (parsed['query_slot'] == 0) == nom_first
        return rm if query_nom else 1-rm
    if method in ('suffix', 'unbound'):
        if parsed['version'] == 'binding':
            return .5
        return oracle_from_local_suffix(tokens, ro, rm)
    raise ValueError(f'Unknown baseline: {method}')
