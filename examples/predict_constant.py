"""Minimal external model adapter: replace constant_probability with your model output.

Usage: python examples/predict_constant.py binding/data/ro075_rm075/test.jsonl predictions.csv
The adapter reads only record_id/tokens, never labels, reliability or oracle metadata.
"""
import csv
import json
from pathlib import Path
import sys


def constant_probability(tokens):
    if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
        raise ValueError('Expected a token list')
    return 0.5


def main(source, output):
    with Path(source).open() as data, Path(output).open('w', newline='') as sink:
        writer = csv.writer(sink)
        writer.writerow(('record_id', 'probability'))
        for line in data:
            row = json.loads(line)
            writer.writerow((row['record_id'], constant_probability(row['tokens'])))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('Usage: predict_constant.py DATASET.jsonl OUTPUT.csv')
    main(sys.argv[1], sys.argv[2])
