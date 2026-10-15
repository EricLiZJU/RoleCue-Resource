#!/usr/bin/env python3
"""Extract and count conservative active-transitive candidates from UD.

The script reports three nested levels: broad syntactic matches, strict simple
active clauses, and language-specific cue-ready candidates. These are
automatic candidates, not validated Agent/Patient examples.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Token:
    id: int
    form: str
    lemma: str
    upos: str
    feats: dict[str, str]
    head: int
    deprel: str


@dataclass
class Sentence:
    sent_id: str
    text: str
    tokens: list[Token]
    source_file: str
    split: str


LANGUAGE_RULES = {
    "en": {"name": "English-EWT"},
    "zh": {"name": "Chinese-GSD"},
    "ja": {"name": "Japanese-GSD"},
    "ru": {"name": "Russian-GSD"},
}
NOMINAL_POS = {"NOUN", "PROPN", "PRON"}
COMPLEX_RELATIONS = {"ccomp", "xcomp", "csubj", "csubj:pass", "conj", "parataxis"}


def parse_feats(raw: str) -> dict[str, str]:
    if raw == "_":
        return {}
    result = {}
    for field in raw.split("|"):
        if "=" in field:
            key, value = field.split("=", 1)
            result[key] = value
    return result


def iter_conllu(path: Path, split: str) -> Iterable[Sentence]:
    comments: dict[str, str] = {}
    tokens: list[Token] = []
    fallback_id = 0
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                if tokens:
                    fallback_id += 1
                    yield Sentence(
                        comments.get("sent_id", f"{path.stem}:{fallback_id}"),
                        comments.get("text", " ".join(t.form for t in tokens)),
                        tokens,
                        path.name,
                        split,
                    )
                comments, tokens = {}, []
                continue
            if line.startswith("#"):
                if "=" in line:
                    key, value = line[1:].split("=", 1)
                    comments[key.strip()] = value.strip()
                continue
            cols = line.split("\t")
            if len(cols) != 10 or "-" in cols[0] or "." in cols[0]:
                continue
            try:
                token_id, head = int(cols[0]), int(cols[6])
            except ValueError:
                continue
            tokens.append(Token(token_id, cols[1], cols[2], cols[3], parse_feats(cols[5]), head, cols[7]))
        if tokens:
            fallback_id += 1
            yield Sentence(
                comments.get("sent_id", f"{path.stem}:{fallback_id}"),
                comments.get("text", " ".join(t.form for t in tokens)),
                tokens,
                path.name,
                split,
            )


def children(tokens: list[Token], head: int) -> list[Token]:
    return [token for token in tokens if token.head == head]


def particle_forms(tokens: list[Token], nominal_id: int) -> list[str]:
    return [token.form for token in tokens if token.head == nominal_id and token.deprel == "case"]


def has_coordination(tokens: list[Token], ids: set[int]) -> bool:
    return any(
        (token.id in ids and token.deprel == "conj")
        or (token.head in ids and token.deprel in {"conj", "cc"})
        for token in tokens
    )


def cue_ready(language: str, sentence: Sentence, pred: Token, subj: Token, obj: Token) -> tuple[bool, str]:
    tokens = sentence.tokens
    if language == "en":
        ready = subj.id < pred.id < obj.id
        return ready, "simple_SVO" if ready else "non_SVO"
    if language == "zh":
        has_ba_bei = any(t.form in {"把", "被"} or t.lemma in {"把", "被"} for t in tokens)
        ready = subj.id < pred.id < obj.id and not has_ba_bei
        return ready, "simple_SVO" if ready else "non_SVO_or_ba_bei"
    if language == "ja":
        subj_particles = set(particle_forms(tokens, subj.id))
        obj_particles = set(particle_forms(tokens, obj.id))
        ready = "が" in subj_particles and "を" in obj_particles
        note = f"subj_particles={','.join(sorted(subj_particles)) or '_'};obj_particles={','.join(sorted(obj_particles)) or '_'}"
        return ready, note
    if language == "ru":
        subj_case = subj.feats.get("Case", "_")
        obj_case = obj.feats.get("Case", "_")
        return subj_case == "Nom" and obj_case == "Acc", f"subj_case={subj_case};obj_case={obj_case}"
    raise ValueError(language)


def analyze_sentence(language: str, sentence: Sentence) -> tuple[list[dict[str, object]], Counter]:
    tokens = sentence.tokens
    counts: Counter = Counter()
    rows: list[dict[str, object]] = []
    by_id = {token.id: token for token in tokens}
    for pred in tokens:
        if pred.upos != "VERB":
            continue
        direct = children(tokens, pred.id)
        subjects = [t for t in direct if t.deprel == "nsubj"]
        objects = [t for t in direct if t.deprel == "obj"]
        if not (len(subjects) == 1 and len(objects) == 1):
            continue
        counts["broad_candidates"] += 1
        subj, obj = subjects[0], objects[0]
        reasons: list[str] = []
        passive = (
            pred.feats.get("Voice") == "Pass"
            or any(t.deprel in {"aux:pass", "nsubj:pass", "obl:agent"} for t in direct)
            or pred.deprel == "nsubj:pass"
        )
        if passive:
            reasons.append("passive")
        if pred.head != 0 or pred.deprel != "root":
            reasons.append("non_root_predicate")
        if subj.upos not in NOMINAL_POS or obj.upos not in NOMINAL_POS:
            reasons.append("non_nominal_argument")
        if has_coordination(tokens, {pred.id, subj.id, obj.id}):
            reasons.append("coordination")
        if any(t.deprel in COMPLEX_RELATIONS for t in direct):
            reasons.append("complex_clause")
        if any(t.deprel == "expl" for t in direct):
            reasons.append("expletive")
        if subj.id == obj.id or subj.id not in by_id or obj.id not in by_id:
            reasons.append("invalid_argument")
        strict = not reasons
        if strict:
            counts["strict_candidates"] += 1
        else:
            for reason in set(reasons):
                counts[f"excluded_{reason}"] += 1
        ready, cue_note = cue_ready(language, sentence, pred, subj, obj)
        cue_ok = strict and ready
        if cue_ok:
            counts["cue_ready_candidates"] += 1
        order = "-".join(label for _, label in sorted([(subj.id, "S"), (pred.id, "V"), (obj.id, "O")]))
        rows.append({
            "language": language,
            "split": sentence.split,
            "source_file": sentence.source_file,
            "sent_id": sentence.sent_id,
            "text": sentence.text,
            "predicate_id": pred.id,
            "predicate_form": pred.form,
            "predicate_lemma": pred.lemma,
            "subject_id": subj.id,
            "subject_form": subj.form,
            "subject_upos": subj.upos,
            "subject_feats": "|".join(f"{k}={v}" for k, v in sorted(subj.feats.items())) or "_",
            "object_id": obj.id,
            "object_form": obj.form,
            "object_upos": obj.upos,
            "object_feats": "|".join(f"{k}={v}" for k, v in sorted(obj.feats.items())) or "_",
            "order": order,
            "strict_candidate": int(strict),
            "cue_ready_candidate": int(cue_ok),
            "exclusion_reasons": ";".join(reasons),
            "cue_note": cue_note,
            "requires_human_audit": 1,
        })
    return rows, counts


def infer_split(path: Path) -> str:
    for split in ("train", "dev", "test"):
        if f"-ud-{split}.conllu" in path.name:
            return split
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=sorted(LANGUAGE_RULES), required=True)
    parser.add_argument("--treebank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()
    files = sorted(path for path in args.treebank.glob("*.conllu") if not path.name.startswith("._"))
    if not files:
        parser.error(f"No .conllu files found in {args.treebank}")
    args.output.mkdir(parents=True, exist_ok=True)
    totals: Counter = Counter()
    split_counts: dict[str, Counter] = {}
    candidate_rows: list[dict[str, object]] = []
    sentence_count = 0
    for file_path in files:
        split = infer_split(file_path)
        split_counts.setdefault(split, Counter())
        print(f"[{args.language}] scanning {file_path.name} ({split})", flush=True)
        for sentence in iter_conllu(file_path, split):
            sentence_count += 1
            totals["sentences"] += 1
            split_counts[split]["sentences"] += 1
            rows, counts = analyze_sentence(args.language, sentence)
            candidate_rows.extend(rows)
            totals.update(counts)
            split_counts[split].update(counts)
            if sentence_count % args.progress_every == 0:
                print(f"[{args.language}] {sentence_count:,} sentences | broad={totals['broad_candidates']:,} strict={totals['strict_candidates']:,} cue-ready={totals['cue_ready_candidates']:,}", flush=True)

    def unique_sentence_count(rows: list[dict[str, object]]) -> int:
        return len({(row["source_file"], row["sent_id"]) for row in rows})

    totals["broad_candidate_sentences"] = unique_sentence_count(candidate_rows)
    totals["strict_candidate_sentences"] = unique_sentence_count(
        [row for row in candidate_rows if row["strict_candidate"]]
    )
    totals["cue_ready_candidate_sentences"] = unique_sentence_count(
        [row for row in candidate_rows if row["cue_ready_candidate"]]
    )
    for split, counts in split_counts.items():
        split_rows = [row for row in candidate_rows if row["split"] == split]
        counts["broad_candidate_sentences"] = unique_sentence_count(split_rows)
        counts["strict_candidate_sentences"] = unique_sentence_count(
            [row for row in split_rows if row["strict_candidate"]]
        )
        counts["cue_ready_candidate_sentences"] = unique_sentence_count(
            [row for row in split_rows if row["cue_ready_candidate"]]
        )
    csv_path = args.output / f"{args.language}_candidates.csv"
    fieldnames = list(candidate_rows[0]) if candidate_rows else ["language"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidate_rows)
    report = {
        "language": args.language,
        "treebank": LANGUAGE_RULES[args.language]["name"],
        "automatic_screening_only": True,
        "criteria": {
            "broad": "VERB with exactly one direct nsubj and one direct obj",
            "strict": "broad + root active clause + nominal explicit arguments + no target coordination, expletive, or clausal complement/control diagnostic",
            "cue_ready": {
                "en": "strict SVO",
                "zh": "strict SVO without surface BA/BEI markers",
                "ja": "strict with subject particle が and object particle を",
                "ru": "strict with Case=Nom subject and Case=Acc object",
            }[args.language],
            "warning": "UD nsubj/obj are grammatical relations, not guaranteed Agent/Patient roles. Causatives, psych predicates, unaccusatives, ambiguity, acceptability, and transformation validity require human review.",
            "counting": "Fields ending in _candidates count predicate matches; fields ending in _candidate_sentences count unique sentences.",
        },
        "totals": dict(sorted(totals.items())),
        "splits": {key: dict(sorted(value.items())) for key, value in sorted(split_counts.items())},
        "candidate_csv": csv_path.name,
    }
    (args.output / f"{args.language}_statistics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[{args.language}] DONE {sentence_count:,} sentences | broad={totals['broad_candidates']:,} strict={totals['strict_candidates']:,} cue-ready={totals['cue_ready_candidates']:,} | {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
