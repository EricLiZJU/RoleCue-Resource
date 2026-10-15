"""Check a clean Core regeneration against the frozen original data bytes."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(dataset: Path, expected: Path) -> dict:
    reference = json.loads(expected.read_text())
    if reference['schema'] != 'rolecue.core-dataset-checksums.v1':
        raise ValueError('Unknown checksum schema')
    generated = json.loads((dataset / 'manifest.json').read_text())
    if generated['status'] != 'validated' or generated['record_counts'] != reference['record_counts']:
        raise ValueError('Generated manifest status/count mismatch')
    for name, info in reference['files'].items():
        path = dataset / name
        if not path.is_file() or path.stat().st_size != info['bytes'] or digest(path) != info['sha256']:
            raise ValueError(f'Core data differs from the frozen source: {name}')
    return {'status': 'verified', 'files': len(reference['files']),
            'records': generated['record_counts']['main'],
            'original_manifest_sha256': reference['original_manifest_sha256']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--expected', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.dataset, args.expected), indent=2))


if __name__ == '__main__':
    main()
