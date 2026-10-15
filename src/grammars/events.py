"""Semantic inventory, independent of noisy surface realizations."""
import hashlib
import itertools
import random
from dataclasses import asdict, dataclass


def derive_seed(seed, *keys):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, (seed, *keys))).encode()).digest()[:8], 'big')


@dataclass(frozen=True)
class Event:
    predicate: str
    agent: str
    patient: str
    split: str

    @property
    def event_id(self):
        return f'{self.predicate}:{self.agent}>{self.patient}'

    @property
    def group_id(self):
        return ':'.join((self.predicate, *sorted((self.agent, self.patient))))

    def to_dict(self):
        return dict(asdict(self), event_id=self.event_id, group_id=self.group_id)


def sample_inventory(sizes, seed, entities=128, predicates=24, excluded_groups=None):
    if any(n <= 0 or n % 2 for n in sizes.values()):
        raise ValueError('Split sizes must be positive and even for reciprocal role balance')
    nouns = [f'n{i:03d}' for i in range(1, entities + 1)]
    verbs = [f'v{i:02d}' for i in range(1, predicates + 1)]
    if sum(sizes.values()) > entities * (entities - 1) * predicates:
        raise ValueError('Requested inventory exceeds unique event capacity')
    excluded_groups = set(excluded_groups or ())
    pools = {}
    for verb in verbs:
        pools[verb] = [pair for pair in itertools.combinations(nouns, 2)
                       if ':'.join((verb, *pair)) not in excluded_groups]
        random.Random(derive_seed(seed, 'pairs', verb)).shuffle(pools[verb])
    random.Random(derive_seed(seed, 'predicate_order')).shuffle(verbs)
    used = dict.fromkeys(verbs, 0)
    result, index = {}, 0
    for split, count in sizes.items():
        result[split] = []
        for _ in range(count // 2):
            verb = verbs[index % len(verbs)]
            if used[verb] >= len(pools[verb]):
                raise ValueError('Requested inventory exceeds capacity after group exclusions')
            a, b = pools[verb][used[verb]]
            used[verb] += 1
            index += 1
            if derive_seed(seed, 'orientation', verb, a, b) % 2:
                a, b = b, a
            result[split].extend((Event(verb, a, b, split), Event(verb, b, a, split)))
    return result


def stage_inventory(master, sizes):
    if any(n <= 0 or n % 2 or n > len(master[k]) for k, n in sizes.items()):
        raise ValueError('Stage sizes must be even and fit in master splits')
    return {k: master[k][:n] for k, n in sizes.items()}
