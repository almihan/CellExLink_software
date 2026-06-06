from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

LOGGER = logging.getLogger(__name__)
PathLike = str | Path


@dataclass(frozen=True, slots=True)
class EntitySpan:
    """A flat local-offset entity span inside one BioC passage."""

    start: int  # local passage offset
    end: int  # local passage offset, exclusive
    label: str
    ann_id: str = ""
    text: str = ""

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"Invalid entity span: {self.start}..{self.end}")


@dataclass(frozen=True, slots=True)
class PassageRecord:
    """One BioC passage converted to a JSONL/Hugging Face record."""

    record_id: int
    document_id: str
    passage_id: int
    passage_offset: int
    text: str
    entities: list[EntitySpan]


def iter_input_files(paths: Sequence[PathLike]) -> Iterator[Path]:
    """Yield files from a mixed list of files and directories."""
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file():
            yield path
        elif path.is_dir():
            for child in sorted(p for p in path.rglob("*") if p.is_file()):
                yield child
        else:
            raise FileNotFoundError(f"Input path does not exist: {path}")


def spans_overlap(left: EntitySpan, right: EntitySpan) -> bool:
    return max(left.start, right.start) < min(left.end, right.end)


def count_overlapping_pairs(entities: Sequence[EntitySpan]) -> int:
    count = 0
    for index, left in enumerate(entities):
        for right in entities[index + 1 :]:
            if spans_overlap(left, right):
                count += 1
    return count


def first_overlapping_pair(entities: Sequence[EntitySpan]) -> Optional[tuple[EntitySpan, EntitySpan]]:
    for index, left in enumerate(entities):
        for right in entities[index + 1 :]:
            if spans_overlap(left, right):
                return left, right
    return None


def _annotation_locations(annotation: Any) -> list[Any]:
    locations = list(getattr(annotation, "locations", []) or [])
    if locations:
        return locations

    total_span = getattr(annotation, "total_span", None)
    if total_span is None:
        return []

    class _Location:
        def __init__(self, offset: int, length: int) -> None:
            self.offset = offset
            self.length = length

    return [_Location(total_span.offset, total_span.length)]


def extract_entities(passage: Any, *, overlap_policy: str = "last") -> list[EntitySpan]:
    """Extract flat annotation spans from one BioC passage."""
    if overlap_policy not in {"last", "error"}:
        raise ValueError("overlap_policy must be one of: last, error.")

    entities: list[EntitySpan] = []
    passage_text = passage.text or ""
    passage_offset = int(passage.offset or 0)
    passage_end = passage_offset + len(passage_text)

    for annotation in passage.annotations:
        label = str(annotation.infons.get("type", "Unknown"))
        annotation_text = annotation.text or ""
        locations = _annotation_locations(annotation)
        if not locations:
            LOGGER.warning("Annotation %s has no locations and no total span; skipping.", annotation.id)
            continue

        if len(locations) > 1:
            LOGGER.warning(
                "Annotation %s has %d locations; treating each location as a separate flat entity.",
                annotation.id,
                len(locations),
            )

        for location_index, location in enumerate(locations):
            absolute_start = int(location.offset)
            absolute_end = absolute_start + int(location.length)
            if absolute_end <= absolute_start:
                LOGGER.warning(
                    "Skipping zero-length/invalid annotation %s at %d-%d.",
                    annotation.id,
                    absolute_start,
                    absolute_end,
                )
                continue

            if absolute_start < passage_offset or absolute_end > passage_end:
                LOGGER.warning(
                    "Skipping annotation %s outside passage bounds: %d-%d not in [%d, %d).",
                    annotation.id,
                    absolute_start,
                    absolute_end,
                    passage_offset,
                    passage_end,
                )
                continue

            local_start = absolute_start - passage_offset
            local_end = absolute_end - passage_offset
            actual_text = passage_text[local_start:local_end]
            if annotation_text and actual_text and annotation_text != actual_text:
                LOGGER.warning(
                    "Annotation text mismatch at absolute offset %d: expected %r, found %r",
                    absolute_start,
                    annotation_text,
                    actual_text,
                )

            ann_id = f"{annotation.id}:{location_index}" if len(locations) > 1 else str(annotation.id)
            entities.append(
                EntitySpan(
                    start=local_start,
                    end=local_end,
                    label=label,
                    ann_id=ann_id,
                    text=annotation_text or actual_text,
                )
            )

    overlap_count = count_overlapping_pairs(entities)
    if overlap_count:
        if overlap_policy == "error":
            pair = first_overlapping_pair(entities)
            assert pair is not None
            raise ValueError(
                "Overlapping entities found in BioC XML, but overlap_policy='error': "
                f"{pair[0].label!r} at {pair[0].start}-{pair[0].end} overlaps "
                f"{pair[1].label!r} at {pair[1].start}-{pair[1].end}."
            )
        LOGGER.warning(
            "Detected %d overlapping entity pair(s) in passage offset %d. "
            "Training with overlap_policy='last' lets later annotations override earlier ones.",
            overlap_count,
            passage_offset,
        )

    return entities


def iter_passage_records(
    srcs: Sequence[PathLike],
    *,
    include_entities: bool,
    overlap_policy: str = "last",
) -> Iterator[PassageRecord]:
    """Stream BioC passages as JSONL-ready records."""
    import bioc as bioc_lib

    record_id = 0
    for src in iter_input_files(srcs):
        LOGGER.info("Reading BioC XML: %s", src)
        with bioc_lib.biocxml.iterparse(str(src)) as reader:
            _ = reader.get_collection_info()
            for document in reader:
                for passage_index, passage in enumerate(document.passages):
                    text = passage.text or ""
                    if not text:
                        continue
                    entities = extract_entities(passage, overlap_policy=overlap_policy) if include_entities else []
                    yield PassageRecord(
                        record_id=record_id,
                        document_id=str(document.id),
                        passage_id=passage_index,
                        passage_offset=int(passage.offset or 0),
                        text=text,
                        entities=entities,
                    )
                    record_id += 1


def convert_bioc_to_json(
    srcs: Sequence[PathLike],
    dest: PathLike,
    *,
    include_entities: bool = True,
    overlap_policy: str = "last",
) -> int:
    """
    Convert BioC XML files into JSONL records used by the NER component.

    The JSONL schema is:
        id, document_id, passage_id, passage_offset, text, entities
    where entities are local passage offsets.
    """
    output_path = Path(dest)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_records = 0
    with output_path.open("w", encoding="utf-8") as sink:
        for record in iter_passage_records(srcs, include_entities=include_entities, overlap_policy=overlap_policy):
            payload: dict[str, Any] = {
                "id": record.record_id,
                "document_id": record.document_id,
                "passage_id": record.passage_id,
                "passage_offset": record.passage_offset,
                "text": record.text,
            }
            if include_entities:
                payload["entities"] = [
                    {
                        "start": entity.start,
                        "end": entity.end,
                        "label": entity.label,
                        "text": entity.text,
                    }
                    for entity in record.entities
                ]
            sink.write(json.dumps(payload, ensure_ascii=False) + "\n")
            num_records += 1

    LOGGER.info(
        "Wrote %d passage records to %s (include_entities=%s, overlap_policy=%s).",
        num_records,
        output_path,
        include_entities,
        overlap_policy,
    )
    return num_records


def iter_predicted_entities(prediction_entries: Sequence[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    """Yield document-level absolute-offset predicted entities."""
    for entry in prediction_entries:
        document_id = str(entry.get("document_id", ""))
        for entity in entry.get("predicted_entities", []) or []:
            yield {
                "document_id": document_id,
                "label": str(entity["label"]),
                "start": int(entity["start"]),
                "end": int(entity["end"]),
                "text": str(entity.get("text", "")),
            }


def _get_bioc_load_dump() -> tuple[Any, Any]:
    import bioc as bioc_lib

    bioc_load = getattr(bioc_lib, "load", None)
    bioc_dump = getattr(bioc_lib, "dump", None)
    if bioc_load is None or bioc_dump is None:
        bioc_load = bioc_lib.biocxml.load
        bioc_dump = bioc_lib.biocxml.dump
    return bioc_load, bioc_dump


def write_predictions_to_bioc_xml(
    input_xml_path: PathLike,
    output_xml_path: PathLike,
    prediction_entries: Sequence[dict[str, Any]],
) -> None:
    """Write predicted NER annotations back into the original BioC XML structure."""
    import bioc as bioc_lib

    entities_by_doc: dict[str, list[dict[str, Any]]] = {}
    for entity in iter_predicted_entities(prediction_entries):
        entities_by_doc.setdefault(str(entity["document_id"]), []).append(entity)

    for doc_entities in entities_by_doc.values():
        doc_entities.sort(key=lambda item: (int(item["start"]), int(item["end"]), str(item["label"])))

    bioc_load, bioc_dump = _get_bioc_load_dump()

    with Path(input_xml_path).open("r", encoding="utf-8") as handle:
        collection = bioc_load(handle)

    annotation_index = 1
    for document in collection.documents:
        document_entities = list(entities_by_doc.get(str(document.id), []))
        entity_pointer = 0

        for passage in document.passages:
            passage.annotations = []
            passage.relations = []

            passage_text = passage.text or ""
            passage_start = int(passage.offset or 0)
            passage_end = passage_start + len(passage_text)

            while entity_pointer < len(document_entities) and int(document_entities[entity_pointer]["end"]) <= passage_start:
                entity_pointer += 1

            local_pointer = entity_pointer
            while local_pointer < len(document_entities):
                entity = document_entities[local_pointer]
                start = int(entity["start"])
                end = int(entity["end"])

                if start >= passage_end:
                    break

                if passage_start <= start < end <= passage_end:
                    annotation = bioc_lib.BioCAnnotation()
                    annotation.id = f"T{annotation_index}"
                    annotation.infons["type"] = str(entity["label"])

                    relative_start = start - passage_start
                    relative_end = end - passage_start
                    annotation.text = passage_text[relative_start:relative_end] if passage_text else str(entity.get("text", ""))
                    annotation.add_location(bioc_lib.BioCLocation(start, end - start))
                    passage.add_annotation(annotation)
                    annotation_index += 1

                local_pointer += 1

    output_path = Path(output_xml_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        bioc_dump(collection, handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert BioC XML into JSONL passage records for CellExLink NER."
    )
    parser.add_argument("inputs", nargs="+", help="Input BioC XML files or directories.")
    parser.add_argument("--output", required=True, help="Destination JSONL file.")
    parser.add_argument(
        "--mode",
        choices=["train", "predict"],
        default="train",
        help="`train` keeps BioC entities as supervision labels; `predict` exports only text records.",
    )
    parser.add_argument(
        "--overlap-policy",
        choices=["last", "error"],
        default="last",
        help="How train mode handles overlapping spans.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    count = convert_bioc_to_json(
        args.inputs,
        args.output,
        include_entities=args.mode == "train",
        overlap_policy=args.overlap_policy,
    )
    print(f"Wrote {count} passage records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
