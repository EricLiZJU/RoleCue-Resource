"""Fixed SOV, separate case tokens, final query; metadata is never an input."""
import re

QUERY = '<QUERY_AGENT>'
TOKENIZER_ID = 'symbolic_whitespace_v1'
E3_TOKENIZER_ID = 'symbolic_structured_whitespace_v1'
E3_SPECIAL_TOKENS = ['<MAIN>', '</MAIN>', '<REL>', '</REL>', '<MOD>', '<COMP>', '<PAD>']


def realize(event, y, sm):
    if event.agent == event.patient or y not in (-1, 1) or sm not in (-1, 1):
        raise ValueError('Invalid roles or directions')
    first, second = (event.agent, event.patient) if y == 1 else (event.patient, event.agent)
    marks = ('<NOM>', '<ACC>') if sm == 1 else ('<ACC>', '<NOM>')
    return [first, marks[0], second, marks[1], event.predicate, QUERY]


def parse(tokens):
    if (len(tokens) != 6 or tokens[5] != QUERY or tokens[0] == tokens[2]
            or not all(re.fullmatch(r'n\d{3}', tokens[i]) for i in (0, 2))
            or not re.fullmatch(r'v\d{2}', tokens[4])
            or (tokens[1], tokens[3]) not in (('<NOM>', '<ACC>'), ('<ACC>', '<NOM>'))):
        raise ValueError('Malformed symbolic sentence')
    return {'candidate_order': [tokens[0], tokens[2]], 'predicate': tokens[4],
            'marker_direction': 1 if tokens[1] == '<NOM>' else -1}


def model_example(record):
    """Only tokens are inputs; label is the supervised target."""
    if record.get('tokenizer') == 'rolecue_binding_v1':
        from src.binding.task import parse as parse_binding
        parsed = parse_binding(record['tokens'])
        if parsed['version'] != record['version']:
            raise ValueError('Binding task mode mismatch')
        label = record['label_query_is_agent']
        if type(label) is not int or label not in (0, 1):
            raise ValueError('Invalid queried-entity target')
        return {'tokens': list(record['tokens']), 'label': label}
    if record.get('tokenizer') == E3_TOKENIZER_ID:
        parse_structured(record['tokens'], record['candidate_order'])
    else:
        parse(record['tokens'])
    label = record['label_first_is_agent']
    if type(label) is not int or label not in (0, 1):
        raise ValueError('Invalid supervised target')
    return {'tokens': list(record['tokens']), 'label': label}


def parse_structured(tokens, candidates=None):
    """Validate E3 surface form without consulting semantic/oracle metadata."""
    if (not 6 <= len(tokens) <= 128 or tokens[-1] != QUERY
            or tokens.count('<MAIN>') != 1 or tokens.count('</MAIN>') != 1
            or tokens.index('<MAIN>') >= tokens.index('</MAIN>')):
        raise ValueError('Malformed E3 structured sentence')
    nouns = [token for token in tokens if re.fullmatch(r'n\d{3}', token)]
    verbs = [token for token in tokens if re.fullmatch(r'v\d{2}', token)]
    if len(nouns) < 2 or not verbs:
        raise ValueError('E3 sentence lacks arguments or predicate')
    if candidates is not None:
        if len(candidates) != 2 or candidates[0] == candidates[1]:
            raise ValueError('Invalid E3 candidates')
        for candidate in candidates:
            if tokens.count(candidate) != 1:
                raise ValueError('Target candidate must occur exactly once')
    stack = []
    pairs = {'<REL>': '</REL>', '<MAIN>': '</MAIN>'}
    closing = {value: key for key, value in pairs.items()}
    for token in tokens:
        if token in pairs:
            stack.append(token)
        elif token in closing:
            if not stack or stack.pop() != closing[token]:
                raise ValueError('Unbalanced E3 structure tokens')
    if stack:
        raise ValueError('Unclosed E3 structure')
    return {'candidate_order': list(candidates) if candidates else nouns[:2],
            'predicate': verbs[-1], 'length': len(tokens)}
