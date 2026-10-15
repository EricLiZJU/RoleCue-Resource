"""Strict token/target projection. Evaluation metadata never reaches a model."""
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from src.grammars.generator import sha256
from src.grammars.linearizer import model_example, QUERY


def read_records(path, expected_split=None):
    with Path(path).open() as handle:
        rows = [json.loads(line) for line in handle]
    if not rows:
        raise ValueError('Empty input dataset')
    if expected_split and any(r['split'] != expected_split for r in rows):
        raise ValueError('Wrong data split')
    return rows


def load_vocabulary(path):
    vocab = json.loads(Path(path).read_text())
    tokens = vocab['entities'] + vocab['predicates'] + vocab['special_tokens']
    if len(set(tokens)) != len(tokens) or QUERY not in tokens:
        raise ValueError('Invalid frozen vocabulary')
    return {token: index for index, token in enumerate(tokens)}


class RoleDataset(Dataset):
    def __init__(self, rows, vocabulary):
        examples = [model_example(r) for r in rows]
        lengths = [len(r['tokens']) for r in examples]
        if not lengths or min(lengths) < 2:
            raise ValueError('Invalid empty or one-token input')
        try:
            encoded = [torch.tensor([vocabulary[t] for t in r['tokens']], dtype=torch.long)
                       for r in examples]
        except KeyError as exc:
            raise ValueError(f'Unknown token: {exc}') from exc
        # Preserve the original tensor interface for fixed-length v1 data while
        # allowing E3 examples to have different lengths. Padding happens only
        # in the batch collator and is accompanied by exact lengths.
        self.token_ids = torch.stack(encoded) if len(set(lengths)) == 1 else tuple(encoded)
        self.lengths = torch.tensor(lengths, dtype=torch.long)
        self.targets = torch.tensor([r['label'] for r in examples], dtype=torch.float32)
        self.pad_id = vocabulary.get('<PAD>', 0)
        # Deliberately do not retain rows, IDs, reliabilities, or oracle fields.

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.token_ids[index], self.targets[index]


def make_loader(dataset, batch_size, shuffle=False, seed=1):
    def collate(batch):
        sequences, targets = zip(*batch)
        lengths = torch.tensor([len(x) for x in sequences], dtype=torch.long)
        padded = torch.full((len(sequences), int(lengths.max())), dataset.pad_id, dtype=torch.long)
        for i, sequence in enumerate(sequences):
            padded[i, :len(sequence)] = sequence
        return padded, torch.stack(targets), lengths
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(seed), drop_last=False,
                      collate_fn=collate)


def verify_file(path, manifest, root):
    name = str(Path(path).relative_to(root))
    if name not in manifest['files'] or sha256(path) != manifest['files'][name]['sha256']:
        raise ValueError(f'Data fingerprint mismatch: {name}')


def require_disjoint(*splits):
    seen = set()
    for rows in splits:
        groups = {r['event_group_id'] for r in rows}
        if seen & groups:
            raise ValueError('Semantic event group leakage across splits')
        seen.update(groups)
