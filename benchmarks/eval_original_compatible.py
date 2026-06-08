"""Original-compatible CellExLink benchmark evaluator.

This module intentionally mirrors the evaluation logic used by the original
CellExLink repository for Cell Ontology normalization and strict end-to-end
linking.  Keep this file simple and stable: benchmark code should measure the
software, not redefine the task.

Important compatibility details:

* Gold-span NEN uses the original ``gold_mention_normalize`` behavior.  It
  scores concept IDs by passage key and entity type, without requiring span
  coordinates in the tuple.  Therefore the prediction XML used for gold-span
  NEN must preserve the gold ``identifier`` infon so the same gold annotations
  can be iterated while reading the predicted ``<model>_id_0`` fields.
* Strict end-to-end evaluation uses exact BioC location tuples plus the CL ID.
* CellLink uses two iterator settings: exact IDs only and all labels.
* Other datasets use the single-ID iterator.

The implementation uses only the Python standard library so it can run in a
minimal SoftwareX reproduction environment.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Optional, Sequence
from xml.etree import ElementTree as ET

PathLike = str | Path
LocationTuple = tuple[tuple[int, int], ...]

DEFAULT_MODEL_NAME = "CellExLink-Sapbert"

DATASET_CONFIGS: dict[str, dict[str, list]] = {
    "celllink": {
        "entity_type_sets": [
            ["cell_phenotype"],
            ["cell_hetero"],
            ["cell_type"],
            ["cell_phenotype", "cell_hetero"],
        ],
        "iterator_names": ["exactIDsOnly_iterator", "allLabels_iterator"],
    },
    "other": {
        "entity_type_sets": [["cell_type"]],
        "iterator_names": ["singleID_iterator"],
    },
}


@dataclass(slots=True)
class Annotation:
    infons: dict[str, str]
    locations: LocationTuple
    text: str


@dataclass(slots=True)
class Passage:
    offset: int
    infons: dict[str, str]
    annotations: list[Annotation]


@dataclass(slots=True)
class Document:
    id: str
    passages: list[Passage]


@dataclass(slots=True)
class Collection:
    documents: list[Document]


@dataclass(slots=True)
class MetricRow:
    dataset: str
    iterator: str
    entity_types: str
    model_name: str
    topk: str
    score_mode: str
    precision: float
    precision_numerator: int
    precision_denominator: int
    recall: float
    recall_numerator: int
    recall_denominator: int
    f1: float
    gold_file: str
    pred_file: str

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "iterator": self.iterator,
            "entity_types": self.entity_types,
            "model_name": self.model_name,
            "topk": self.topk,
            "score_mode": self.score_mode,
            "precision": self.precision,
            "precision_numerator": self.precision_numerator,
            "precision_denominator": self.precision_denominator,
            "recall": self.recall,
            "recall_numerator": self.recall_numerator,
            "recall_denominator": self.recall_denominator,
            "f1": self.f1,
            "gold_file": self.gold_file,
            "pred_file": self.pred_file,
        }


def _child_text(element: ET.Element, name: str, default: str = "") -> str:
    child = element.find(name)
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


def load_collection(input_path: PathLike) -> Collection:
    """Read a BioC XML collection into a small evaluator data model."""

    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    root = ET.parse(path).getroot()
    documents: list[Document] = []

    for doc_el in root.findall("document"):
        doc_id = _child_text(doc_el, "id", "")
        passages: list[Passage] = []

        for passage_el in doc_el.findall("passage"):
            passage_offset = _safe_int(_child_text(passage_el, "offset", "0"), 0)
            passage_infons = _infons(passage_el)
            annotations: list[Annotation] = []

            for ann_el in passage_el.findall("annotation"):
                ann_infons = _infons(ann_el)
                locations: list[tuple[int, int]] = []
                for loc_el in ann_el.findall("location"):
                    offset = _safe_int(loc_el.attrib.get("offset"), -1)
                    length = _safe_int(loc_el.attrib.get("length"), -1)
                    if offset >= 0 and length > 0:
                        locations.append((offset, length))
                annotations.append(
                    Annotation(
                        infons=ann_infons,
                        locations=tuple(sorted(locations)),
                        text=_child_text(ann_el, "text", "").strip(),
                    )
                )

            passages.append(
                Passage(
                    offset=passage_offset,
                    infons=passage_infons,
                    annotations=annotations,
                )
            )

        documents.append(Document(id=doc_id, passages=passages))

    return Collection(documents=documents)


def get_passage_key(doc: Document, passage: Passage) -> str:
    """Return the original CellExLink passage key."""

    passage_id = passage.infons.get("passage_id")
    if passage_id not in (None, ""):
        return passage_id

    article_id = passage.infons.get("article-id_pmid")
    if article_id not in (None, ""):
        return article_id

    return f"{doc.id}:{passage.offset}"


def clean_identifier_text(identifier_text: str) -> str:
    return (
        identifier_text.strip()
        .replace("(skos:exact)", "")
        .replace("(skos:related)", "")
        .strip()
    )


def split_identifier_field(identifier_text: str) -> list[str]:
    cleaned = clean_identifier_text(identifier_text)
    parts = [part.strip() for part in re.split(r"[;,]", cleaned)]
    return [part for part in parts if part and part != "-" and part.lower() != "none"]


def exact_ids_only_iterator(
    collection: Collection,
) -> Iterator[tuple[Document, Passage, Annotation, tuple[str, ...]]]:
    """CellLink exact-ID iterator from the original evaluator."""

    for doc in collection.documents:
        for passage in doc.passages:
            for ann in passage.annotations:
                identifier = ann.infons.get("identifier")
                if not identifier:
                    continue
                text = identifier.strip()
                if not text or "none" in text.lower():
                    continue
                if ";" in text or "," in text:
                    continue
                if "(skos:related)" in text.lower():
                    continue
                normalized_id = clean_identifier_text(text)
                if normalized_id and normalized_id != "-":
                    yield doc, passage, ann, (normalized_id,)


def single_id_iterator(
    collection: Collection,
) -> Iterator[tuple[Document, Passage, Annotation, tuple[str, ...]]]:
    """Single-ID iterator used for CRAFT, BioID, and other non-CellLink data."""

    for doc in collection.documents:
        for passage in doc.passages:
            for ann in passage.annotations:
                identifier = ann.infons.get("identifier")
                if not identifier:
                    continue
                text = identifier.strip()
                if not text or "none" in text.lower():
                    continue
                if ";" in text or "," in text:
                    continue
                normalized_id = clean_identifier_text(text)
                if normalized_id and normalized_id != "-":
                    yield doc, passage, ann, (normalized_id,)


def all_labels_iterator(
    collection: Collection,
) -> Iterator[tuple[Document, Passage, Annotation, tuple[str, ...]]]:
    """CellLink all-label iterator from the original evaluator."""

    for doc in collection.documents:
        for passage in doc.passages:
            for ann in passage.annotations:
                identifier = ann.infons.get("identifier")
                if not identifier:
                    continue
                all_ids = split_identifier_field(identifier)
                if all_ids:
                    yield doc, passage, ann, tuple(all_ids)


ITERATORS = {
    "exactIDsOnly_iterator": exact_ids_only_iterator,
    "singleID_iterator": single_id_iterator,
    "allLabels_iterator": all_labels_iterator,
}


def build_score_key(model_name: str, rank: int) -> str:
    return f"{model_name}_identifier_score_{rank}"


def get_prediction_tuples(
    doc: Document,
    passage: Passage,
    ann: Annotation,
    model_name: str,
    *,
    max_k: int = 10,
    include_locations: bool = False,
    score_threshold: Optional[float] = None,
) -> list[tuple]:
    """Read top-k prediction tuples from one annotation."""

    tuples: list[tuple] = []
    passage_key = get_passage_key(doc, passage)
    ann_type = ann.infons.get("type", "")
    locations = ann.locations

    for rank in range(max_k):
        id_key = f"{model_name}_id_{rank}"
        if id_key not in ann.infons:
            continue

        pred_id = (ann.infons[id_key] or "").strip()
        if pred_id.lower() in {"", "-", "none"}:
            continue

        if score_threshold is not None:
            score_text = ann.infons.get(build_score_key(model_name, rank))
            if score_text in (None, ""):
                continue
            try:
                if float(score_text) < score_threshold:
                    continue
            except ValueError:
                continue

        if include_locations:
            tuples.append((passage_key, ann_type, locations, pred_id))
        else:
            tuples.append((passage_key, ann_type, pred_id))

    return tuples


def get_reference_tuples(
    reference_collection: Collection,
    iterator_name: str,
    entity_types: Sequence[str],
    *,
    include_locations: bool = False,
) -> set[tuple]:
    iterator = ITERATORS[iterator_name]
    entity_type_set = set(entity_types)
    ref_tuples: set[tuple] = set()

    for doc, passage, ann, reference_ids in iterator(reference_collection):
        ann_type = ann.infons.get("type")
        if ann_type not in entity_type_set:
            continue

        passage_key = get_passage_key(doc, passage)
        if include_locations:
            locations = ann.locations
            ref_tuples.update(
                (passage_key, ann_type, locations, reference_id)
                for reference_id in reference_ids
            )
        else:
            ref_tuples.update(
                (passage_key, ann_type, reference_id) for reference_id in reference_ids
            )

    return ref_tuples


def iter_prediction_annotations(
    prediction_collection: Collection,
    entity_types: Sequence[str],
) -> Iterator[tuple[Document, Passage, Annotation]]:
    entity_type_set = set(entity_types)
    for doc in prediction_collection.documents:
        for passage in doc.passages:
            for ann in passage.annotations:
                if ann.infons.get("type") in entity_type_set:
                    yield doc, passage, ann


def _metrics(reference_tuples: set[tuple], prediction_tuples: set[tuple]) -> dict[str, float | int]:
    matches = len(reference_tuples.intersection(prediction_tuples))
    precision_denominator = len(prediction_tuples)
    recall_denominator = len(reference_tuples)
    precision = matches / precision_denominator if precision_denominator else 0.0
    recall = matches / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "precision_numerator": matches,
        "precision_denominator": precision_denominator,
        "recall": recall,
        "recall_numerator": matches,
        "recall_denominator": recall_denominator,
        "f1": f1,
    }


def evaluate_iterator(
    *,
    reference_collection: Collection,
    prediction_collection: Collection,
    iterator_name: str,
    model_names: Sequence[str],
    entity_types: Sequence[str],
    score_mode: str,
    score_threshold: Optional[float] = 0.0,
    max_k: int = 10,
) -> list[dict[str, object]]:
    """Evaluate one iterator/entity-type setting.

    ``score_mode`` must be ``gold_mention_normalize`` or ``end_to_end``.
    """

    if score_mode not in {"gold_mention_normalize", "end_to_end"}:
        raise ValueError("score_mode must be 'gold_mention_normalize' or 'end_to_end'")

    include_locations = score_mode == "end_to_end"
    results: list[dict[str, object]] = []

    for model_name in model_names:
        reference_tuples = get_reference_tuples(
            reference_collection,
            iterator_name,
            entity_types,
            include_locations=include_locations,
        )

        top1_prediction_tuples: set[tuple] = set()
        top5_prediction_tuples: set[tuple] = set()
        top10_prediction_tuples: set[tuple] = set()

        if include_locations:
            prediction_iterator = iter_prediction_annotations(prediction_collection, entity_types)
            for doc, passage, ann in prediction_iterator:
                top10 = get_prediction_tuples(
                    doc,
                    passage,
                    ann,
                    model_name,
                    max_k=max_k,
                    include_locations=True,
                    score_threshold=score_threshold,
                )
                if not top10:
                    continue
                top1_prediction_tuples.add(top10[0])
                top5_prediction_tuples.update(top10[:5])
                top10_prediction_tuples.update(top10)
        else:
            # Original gold-span NEN behavior: iterate prediction annotations with
            # the same gold-ID iterator. This means the prediction XML must keep
            # each annotation's gold ``identifier`` infon.
            iterator = ITERATORS[iterator_name]
            for doc, passage, ann, _ in iterator(prediction_collection):
                if ann.infons.get("type") not in set(entity_types):
                    continue
                top10 = get_prediction_tuples(
                    doc,
                    passage,
                    ann,
                    model_name,
                    max_k=max_k,
                    include_locations=False,
                    score_threshold=score_threshold,
                )
                if not top10:
                    continue
                top1_prediction_tuples.add(top10[0])
                top5_prediction_tuples.update(top10[:5])
                top10_prediction_tuples.update(top10)

        topk_sets = [
            ("1", top1_prediction_tuples),
            ("5", top5_prediction_tuples),
            ("10", top10_prediction_tuples),
        ]
        model_result = {"model_name": model_name, "topk": []}
        for k, prediction_set in topk_sets:
            metric = _metrics(reference_tuples, prediction_set)
            model_result["topk"].append(
                {
                    "k": k,
                    "score_mode": score_mode,
                    **metric,
                }
            )
        results.append(model_result)

    return results


def evaluate_file_pair(
    *,
    dataset_name: str,
    gold_file: PathLike,
    pred_file: PathLike,
    dataset_style: str,
    score_mode: str,
    model_names: Sequence[str] = (DEFAULT_MODEL_NAME,),
    topk: str = "1",
    score_threshold: Optional[float] = 0.0,
) -> list[MetricRow]:
    if dataset_style not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset style: {dataset_style}")
    if topk not in {"1", "5", "10", "all"}:
        raise ValueError("topk must be one of: 1, 5, 10, all")

    reference_collection = load_collection(gold_file)
    prediction_collection = load_collection(pred_file)
    config = DATASET_CONFIGS[dataset_style]
    rows: list[MetricRow] = []

    for iterator_name in config["iterator_names"]:
        for entity_types in config["entity_type_sets"]:
            results = evaluate_iterator(
                reference_collection=reference_collection,
                prediction_collection=prediction_collection,
                iterator_name=iterator_name,
                model_names=model_names,
                entity_types=entity_types,
                score_mode=score_mode,
                score_threshold=score_threshold,
            )
            for model_result in results:
                for topk_result in model_result["topk"]:
                    if topk != "all" and topk_result["k"] != topk:
                        continue
                    rows.append(
                        MetricRow(
                            dataset=dataset_name,
                            iterator=iterator_name,
                            entity_types="+".join(entity_types),
                            model_name=str(model_result["model_name"]),
                            topk=str(topk_result["k"]),
                            score_mode=str(topk_result["score_mode"]),
                            precision=float(topk_result["precision"]),
                            precision_numerator=int(topk_result["precision_numerator"]),
                            precision_denominator=int(topk_result["precision_denominator"]),
                            recall=float(topk_result["recall"]),
                            recall_numerator=int(topk_result["recall_numerator"]),
                            recall_denominator=int(topk_result["recall_denominator"]),
                            f1=float(topk_result["f1"]),
                            gold_file=str(gold_file),
                            pred_file=str(pred_file),
                        )
                    )

    return rows


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


def pair_named_paths(
    gold_values: Sequence[str],
    pred_values: Sequence[str],
) -> list[tuple[str, Path, Path]]:
    gold = parse_named_paths(gold_values)
    pred = parse_named_paths(pred_values)
    if set(gold) != set(pred):
        raise ValueError(
            f"Dataset names differ: gold={sorted(gold)}, pred={sorted(pred)}"
        )
    return [(name, gold[name], pred[name]) for name in sorted(gold)]


def write_csv(rows: Iterable[MetricRow], output_csv: PathLike) -> Path:
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    row_dicts = [row.as_dict() for row in rows]
    fieldnames = list(MetricRow.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row_dicts)
    return path


def print_rows(rows: Sequence[MetricRow]) -> None:
    for row in rows:
        print(
            f"{row.dataset}\t{row.iterator}\t{row.entity_types}\t"
            f"{row.score_mode}\ttop-{row.topk}\t"
            f"P={row.precision:.3f} ({row.precision_numerator}/{row.precision_denominator})\t"
            f"R={row.recall:.3f} ({row.recall_numerator}/{row.recall_denominator})\t"
            f"F1={row.f1:.3f}"
        )


def build_common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--gold", action="append", required=True, help="Gold XML as DATASET=path. Repeatable.")
    parser.add_argument("--pred", action="append", required=True, help="Prediction XML as DATASET=path. Repeatable.")
    parser.add_argument(
        "--dataset-style",
        choices=sorted(DATASET_CONFIGS),
        default="other",
        help="Use 'celllink' for CellLink and 'other' for CRAFT/BioID.",
    )
    parser.add_argument("--model-names", nargs="+", default=[DEFAULT_MODEL_NAME])
    parser.add_argument("--topk", choices=["1", "5", "10", "all"], default="1")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser
