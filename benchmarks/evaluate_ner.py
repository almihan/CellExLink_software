#!/usr/bin/env python
"""Evaluate CellExLink named entity recognition with exact and relaxed spans.

This is the minimal SoftwareX benchmark script for the NER part of CellExLink.
It intentionally scores only mention detection, not Cell Ontology identifiers.

Metrics
-------
* exact_span: a prediction is correct only when the document/passage key and
  BioC location tuple match exactly.
* relaxed_span: a prediction is correct when it overlaps a gold mention in the
  same passage.  Generic predictions such as "cell", "cells", and "cellular"
  are blacklisted from relaxed matching, so they remain false positives rather
  than receiving partial-match credit.

The output is micro precision/recall/F1 per dataset.  With --macro-average, the
script also writes a macro-average row computed as the arithmetic mean of the
per-dataset precision/recall/F1 values for each criterion.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from xml.etree import ElementTree as ET

PathLike = str | Path
LocationTuple = tuple[tuple[int, int], ...]

DEFAULT_SYSTEM_NAME = "CellExLink"
DEFAULT_INCLUDE_TYPES = ("cell_type", "cell_phenotype", "cell_hetero")
DEFAULT_EXCLUDE_TYPES = ("cell_vague",)
DEFAULT_RELAXED_BLACKLIST = ("cell", "cells", "cellular")

TOKEN_RE = re.compile(r"\S+")
SPACE_RE = re.compile(r"\s+")
PUNCT_EDGE_RE = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)


@dataclass(frozen=True, slots=True)
class Mention:
    document_id: str
    passage_key: str
    passage_offset: int
    passage_text: str
    locations: LocationTuple
    text: str
    entity_type: str

    @property
    def exact_key(self) -> tuple[str, str, LocationTuple]:
        return (self.document_id, self.passage_key, self.locations)

    @property
    def intervals(self) -> tuple[tuple[int, int], ...]:
        return tuple((offset, offset + length) for offset, length in self.locations)

    @property
    def normalized_text(self) -> str:
        return normalize_mention_text(self.text)


@dataclass(frozen=True, slots=True)
class Metrics:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def as_row(
        self,
        *,
        dataset: str,
        system: str,
        criterion: str,
        gold_file: str,
        pred_file: str,
    ) -> dict[str, object]:
        return {
            "dataset": dataset,
            "system": system,
            "task": "NER",
            "criterion": criterion,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "gold_file": gold_file,
            "pred_file": pred_file,
        }


def _child_text(element: ET.Element, child_name: str, default: str = "") -> str:
    child = element.find(child_name)
    if child is None or child.text is None:
        return default
    return child.text


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def _infons(element: ET.Element) -> dict[str, str]:
    out: dict[str, str] = {}
    for infon in element.findall("infon"):
        key = infon.attrib.get("key", "")
        if key:
            out[key] = infon.text or ""
    return out


def get_passage_key(document_id: str, passage_offset: int, passage_infons: Mapping[str, str]) -> str:
    """Use the same stable passage key convention as the normalization evaluator."""

    passage_id = passage_infons.get("passage_id")
    if passage_id not in (None, ""):
        return str(passage_id)

    article_id = passage_infons.get("article-id_pmid")
    if article_id not in (None, ""):
        return str(article_id)

    return f"{document_id}:{passage_offset}"


def normalize_mention_text(text: str) -> str:
    normalized = SPACE_RE.sub(" ", (text or "").strip().lower())
    # Keep internal characters such as CD8+ but ignore edge punctuation for the
    # relaxed blacklist check, e.g. "cells," should be treated as "cells".
    normalized = PUNCT_EDGE_RE.sub("", normalized)
    return normalized


def _type_is_allowed(entity_type: str, include_types: set[str], exclude_types: set[str]) -> bool:
    lower = (entity_type or "cell_type").lower()
    if lower in exclude_types:
        return False
    if include_types and lower not in include_types:
        return False
    return True


def load_bioc_mentions(
    path: PathLike,
    *,
    include_types: Iterable[str] = DEFAULT_INCLUDE_TYPES,
    exclude_types: Iterable[str] = DEFAULT_EXCLUDE_TYPES,
) -> list[Mention]:
    """Load cell-type mention annotations from a BioC XML file."""

    xml_path = Path(path)
    if not xml_path.is_file():
        raise FileNotFoundError(xml_path)

    include_type_set = {value.lower() for value in include_types if value}
    exclude_type_set = {value.lower() for value in exclude_types if value}

    root = ET.parse(xml_path).getroot()
    mentions: list[Mention] = []

    for document_el in root.findall("document"):
        document_id = _child_text(document_el, "id", "")

        for passage_el in document_el.findall("passage"):
            passage_offset = _safe_int(_child_text(passage_el, "offset", "0"), 0)
            passage_text = _child_text(passage_el, "text", "")
            passage_infons = _infons(passage_el)
            passage_key = get_passage_key(document_id, passage_offset, passage_infons)

            for ann_el in passage_el.findall("annotation"):
                infons = _infons(ann_el)
                entity_type = (
                    infons.get("type")
                    or infons.get("entity_type")
                    or infons.get("label")
                    or "cell_type"
                )
                if not _type_is_allowed(entity_type, include_type_set, exclude_type_set):
                    continue

                locations: list[tuple[int, int]] = []
                for loc_el in ann_el.findall("location"):
                    offset = _safe_int(loc_el.attrib.get("offset"), -1)
                    length = _safe_int(loc_el.attrib.get("length"), -1)
                    if offset >= 0 and length > 0:
                        locations.append((offset, length))
                if not locations:
                    continue

                ann_text = _child_text(ann_el, "text", "").strip()
                if not ann_text and len(locations) == 1 and passage_text:
                    offset, length = locations[0]
                    local_start = offset - passage_offset
                    local_end = local_start + length
                    if 0 <= local_start <= local_end <= len(passage_text):
                        ann_text = passage_text[local_start:local_end]

                mentions.append(
                    Mention(
                        document_id=document_id,
                        passage_key=passage_key,
                        passage_offset=passage_offset,
                        passage_text=passage_text,
                        locations=tuple(sorted(locations)),
                        text=ann_text,
                        entity_type=entity_type,
                    )
                )

    return mentions


def _intervals_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) < min(a[1], b[1])


def _char_overlap(gold: Mention, pred: Mention) -> int:
    overlap = 0
    for g_interval in gold.intervals:
        for p_interval in pred.intervals:
            start = max(g_interval[0], p_interval[0])
            end = min(g_interval[1], p_interval[1])
            if start < end:
                overlap += end - start
    return overlap


def _token_intervals_for_passage(passage_offset: int, passage_text: str) -> list[tuple[int, int]]:
    return [
        (passage_offset + match.start(), passage_offset + match.end())
        for match in TOKEN_RE.finditer(passage_text or "")
    ]


def _mention_token_indices(mention: Mention) -> set[int]:
    if not mention.passage_text:
        return set()
    token_intervals = _token_intervals_for_passage(mention.passage_offset, mention.passage_text)
    indices: set[int] = set()
    for idx, token_interval in enumerate(token_intervals):
        if any(_intervals_overlap(token_interval, mention_interval) for mention_interval in mention.intervals):
            indices.add(idx)
    return indices


def mentions_relaxed_overlap(gold: Mention, pred: Mention) -> bool:
    if gold.document_id != pred.document_id or gold.passage_key != pred.passage_key:
        return False

    # Prefer token-level overlap when passage text is available for both sides.
    gold_tokens = _mention_token_indices(gold)
    pred_tokens = _mention_token_indices(pred)
    if gold_tokens and pred_tokens:
        return bool(gold_tokens.intersection(pred_tokens))

    # Fallback for minimal BioC files without passage text.
    return _char_overlap(gold, pred) > 0


def evaluate_exact(gold_mentions: Sequence[Mention], pred_mentions: Sequence[Mention]) -> Metrics:
    """One-to-one exact span matching."""

    lookup: dict[tuple[str, str, LocationTuple], list[int]] = {}
    for idx, gold in enumerate(gold_mentions):
        lookup.setdefault(gold.exact_key, []).append(idx)

    matched_gold: set[int] = set()
    matched_pred: set[int] = set()

    for pred_idx, pred in enumerate(pred_mentions):
        for gold_idx in lookup.get(pred.exact_key, []):
            if gold_idx not in matched_gold:
                matched_gold.add(gold_idx)
                matched_pred.add(pred_idx)
                break

    return Metrics(
        tp=len(matched_gold),
        fp=len(pred_mentions) - len(matched_pred),
        fn=len(gold_mentions) - len(matched_gold),
    )


def evaluate_relaxed(
    gold_mentions: Sequence[Mention],
    pred_mentions: Sequence[Mention],
    *,
    relaxed_blacklist: Iterable[str] = DEFAULT_RELAXED_BLACKLIST,
) -> Metrics:
    """Greedy one-to-one relaxed span matching with generic-term blacklist."""

    blacklist = {normalize_mention_text(value) for value in relaxed_blacklist if value}
    matched_gold: set[int] = set()
    matched_pred: set[int] = set()

    for pred_idx, pred in enumerate(pred_mentions):
        if pred.normalized_text in blacklist:
            # The prediction remains a false positive if it is not exact-matched
            # elsewhere; it is simply not allowed to claim relaxed credit.
            continue

        best_gold_idx: int | None = None
        best_overlap = -1
        for gold_idx, gold in enumerate(gold_mentions):
            if gold_idx in matched_gold:
                continue
            if not mentions_relaxed_overlap(gold, pred):
                continue
            overlap = _char_overlap(gold, pred)
            if overlap > best_overlap:
                best_gold_idx = gold_idx
                best_overlap = overlap

        if best_gold_idx is not None:
            matched_gold.add(best_gold_idx)
            matched_pred.add(pred_idx)

    return Metrics(
        tp=len(matched_gold),
        fp=len(pred_mentions) - len(matched_pred),
        fn=len(gold_mentions) - len(matched_gold),
    )


def parse_named_paths(values: Sequence[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values:
        if "=" in value:
            name, path_text = value.split("=", 1)
            out[name.strip()] = Path(path_text)
        else:
            path = Path(value)
            out[path.stem] = path
    return out


def pair_named_paths(gold_values: Sequence[str], pred_values: Sequence[str]) -> list[tuple[str, Path, Path]]:
    gold = parse_named_paths(gold_values)
    pred = parse_named_paths(pred_values)
    if set(gold) != set(pred):
        raise ValueError(f"Dataset names differ: gold={sorted(gold)}, pred={sorted(pred)}")
    return [(name, gold[name], pred[name]) for name in sorted(gold)]


def evaluate_file_pair(
    *,
    dataset: str,
    gold_file: PathLike,
    pred_file: PathLike,
    system: str = DEFAULT_SYSTEM_NAME,
    include_types: Iterable[str] = DEFAULT_INCLUDE_TYPES,
    exclude_types: Iterable[str] = DEFAULT_EXCLUDE_TYPES,
    relaxed_blacklist: Iterable[str] = DEFAULT_RELAXED_BLACKLIST,
) -> list[dict[str, object]]:
    gold_mentions = load_bioc_mentions(
        gold_file,
        include_types=include_types,
        exclude_types=exclude_types,
    )
    pred_mentions = load_bioc_mentions(
        pred_file,
        include_types=include_types,
        exclude_types=exclude_types,
    )

    exact = evaluate_exact(gold_mentions, pred_mentions)
    relaxed = evaluate_relaxed(
        gold_mentions,
        pred_mentions,
        relaxed_blacklist=relaxed_blacklist,
    )

    return [
        exact.as_row(
            dataset=dataset,
            system=system,
            criterion="exact_span",
            gold_file=str(gold_file),
            pred_file=str(pred_file),
        ),
        relaxed.as_row(
            dataset=dataset,
            system=system,
            criterion="relaxed_span",
            gold_file=str(gold_file),
            pred_file=str(pred_file),
        ),
    ]


def add_macro_average_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Append macro-average rows over datasets for each system/criterion pair."""

    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        if str(row.get("dataset")) == "Macro average":
            continue
        key = (str(row.get("system")), str(row.get("task")), str(row.get("criterion")))
        grouped.setdefault(key, []).append(row)

    macro_rows: list[dict[str, object]] = []
    for (system, task, criterion), group_rows in sorted(grouped.items()):
        if not group_rows:
            continue
        macro = {
            "dataset": "Macro average",
            "system": system,
            "task": task,
            "criterion": criterion,
            "tp": "",
            "fp": "",
            "fn": "",
            "precision": sum(float(row["precision"]) for row in group_rows) / len(group_rows),
            "recall": sum(float(row["recall"]) for row in group_rows) / len(group_rows),
            "f1": sum(float(row["f1"]) for row in group_rows) / len(group_rows),
            "gold_file": "",
            "pred_file": "",
        }
        macro_rows.append(macro)

    return list(rows) + macro_rows


def write_csv(rows: Sequence[Mapping[str, object]], output_csv: PathLike) -> Path:
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "system",
        "task",
        "criterion",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "gold_file",
        "pred_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def print_rows(rows: Sequence[Mapping[str, object]]) -> None:
    for row in rows:
        print(
            f"{row['system']}\t{row['dataset']}\t{row['criterion']}\t"
            f"P={float(row['precision']):.3f}\t"
            f"R={float(row['recall']):.3f}\t"
            f"F1={float(row['f1']):.3f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate CellExLink NER predictions with exact and relaxed span matching."
    )
    parser.add_argument(
        "--gold",
        action="append",
        required=True,
        help="Gold BioC XML as DATASET=path or path. Repeatable.",
    )
    parser.add_argument(
        "--pred",
        action="append",
        required=True,
        help="Prediction BioC XML as DATASET=path or path. Repeatable.",
    )
    parser.add_argument("--system", default=DEFAULT_SYSTEM_NAME)
    parser.add_argument(
        "--include-type",
        action="append",
        default=list(DEFAULT_INCLUDE_TYPES),
        help=(
            "Annotation type to include. Repeatable. Defaults to cell_type, "
            "cell_phenotype, and cell_hetero."
        ),
    )
    parser.add_argument(
        "--exclude-type",
        action="append",
        default=list(DEFAULT_EXCLUDE_TYPES),
        help="Annotation type to exclude. Repeatable. Default: cell_vague.",
    )
    parser.add_argument(
        "--relaxed-blacklist",
        nargs="*",
        default=list(DEFAULT_RELAXED_BLACKLIST),
        help="Generic prediction texts that cannot receive relaxed-match credit.",
    )
    parser.add_argument(
        "--macro-average",
        action="store_true",
        help="Add macro-average rows across datasets for exact and relaxed F1.",
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    rows: list[dict[str, object]] = []
    for dataset, gold_file, pred_file in pair_named_paths(args.gold, args.pred):
        rows.extend(
            evaluate_file_pair(
                dataset=dataset,
                gold_file=gold_file,
                pred_file=pred_file,
                system=args.system,
                include_types=args.include_type,
                exclude_types=args.exclude_type,
                relaxed_blacklist=args.relaxed_blacklist,
            )
        )

    if args.macro_average:
        rows = add_macro_average_rows(rows)

    write_csv(rows, args.output_csv)
    if not args.quiet:
        print_rows(rows)
        print(f"Wrote NER results to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
