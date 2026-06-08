"""Lightweight BioC XML and JSONL I/O utilities for CellExLink.

The functions in this module intentionally avoid importing machine-learning
libraries. They are used by tests, examples, and benchmark wrappers to create
small BioC files, read passages/annotations, and write predicted entities back
into BioC XML.

Offsets
-------
BioC annotation ``location`` offsets are absolute document offsets.  Therefore
``PredictedEntity.start`` and ``PredictedEntity.end`` are also treated as
absolute offsets when writing prediction XML.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence
from xml.etree import ElementTree as ET

PathLike = str | os.PathLike[str]


@dataclass(slots=True)
class PredictedEntity:
    """A predicted entity with absolute BioC offsets."""

    document_id: str = ""
    passage_id: int = 0
    start: int = 0
    end: int = 0
    label: str = "cell_type"
    text: str = ""
    score: float | None = None
    infons: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return max(0, int(self.end) - int(self.start))


@dataclass(slots=True)
class EntitySpan:
    """An entity span relative to a passage unless converted to absolute."""

    start: int
    end: int
    label: str = "cell_type"
    text: str = ""
    score: float | None = None
    infons: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return max(0, int(self.end) - int(self.start))

    def to_absolute(
        self,
        passage_offset: int,
        *,
        document_id: str = "",
        passage_id: int = 0,
    ) -> PredictedEntity:
        """Return a :class:`PredictedEntity` using absolute BioC offsets."""

        offset = int(passage_offset)
        return PredictedEntity(
            document_id=document_id,
            passage_id=passage_id,
            start=offset + int(self.start),
            end=offset + int(self.end),
            label=self.label,
            text=self.text,
            score=self.score,
            infons=dict(self.infons),
        )


@dataclass(slots=True)
class PassageRecord:
    """A BioC passage and its optional entity annotations."""

    record_id: int
    document_id: str
    passage_id: int
    passage_offset: int
    text: str
    entities: list[EntitySpan] = field(default_factory=list)
    infons: dict[str, str] = field(default_factory=dict)
    source_path: str = ""


def _coerce_paths(paths: PathLike | Sequence[PathLike]) -> list[Path]:
    if isinstance(paths, (str, os.PathLike)):
        return [Path(paths)]
    return [Path(path) for path in paths]


def _text_of(parent: ET.Element, tag: str, default: str = "") -> str:
    child = parent.find(tag)
    if child is None or child.text is None:
        return default
    return child.text


def _int_text_of(parent: ET.Element, tag: str, default: int = 0) -> int:
    value = _text_of(parent, tag, str(default)).strip()
    try:
        return int(value)
    except ValueError:
        return default


def _indent_tree(root: ET.Element) -> None:
    # ``ET.indent`` was added in Python 3.9. CellExLink supports Python >=3.10,
    # so this is safe and keeps generated XML readable.
    ET.indent(root, space="  ")


def read_infons(element: ET.Element) -> dict[str, str]:
    """Read BioC ``infon`` children from an XML element."""

    infons: dict[str, str] = {}
    for infon in element.findall("infon"):
        key = infon.attrib.get("key", "")
        if not key:
            continue
        infons[key] = infon.text or ""
    return infons


def write_text_as_bioc_xml(
    text: str,
    output_xml: PathLike,
    *,
    document_id: str = "doc0",
    passage_offset: int = 0,
    source: str = "CellExLink",
    key: str = "cell-type-extraction",
    passage_type: str | None = None,
) -> Path:
    """Write plain text as a one-document, one-passage BioC XML file."""

    output_path = Path(output_xml)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    collection = ET.Element("collection")
    ET.SubElement(collection, "source").text = source
    ET.SubElement(collection, "date").text = ""
    ET.SubElement(collection, "key").text = key

    document = ET.SubElement(collection, "document")
    ET.SubElement(document, "id").text = document_id

    passage = ET.SubElement(document, "passage")
    if passage_type is not None:
        ET.SubElement(passage, "infon", {"key": "type"}).text = passage_type
    ET.SubElement(passage, "offset").text = str(int(passage_offset))
    ET.SubElement(passage, "text").text = text

    _indent_tree(collection)
    ET.ElementTree(collection).write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def _annotation_to_entity_span(annotation: ET.Element, passage_offset: int) -> EntitySpan:
    infons = read_infons(annotation)
    label = infons.get("type") or infons.get("label") or ""
    text = _text_of(annotation, "text", "")

    location = annotation.find("location")
    if location is None:
        absolute_start = passage_offset
        length = len(text)
    else:
        absolute_start = int(location.attrib.get("offset", passage_offset))
        length = int(location.attrib.get("length", len(text)))

    relative_start = absolute_start - int(passage_offset)
    relative_end = relative_start + length

    score: float | None = None
    if "score" in infons:
        try:
            score = float(infons["score"])
        except ValueError:
            score = None

    return EntitySpan(
        start=relative_start,
        end=relative_end,
        label=label,
        text=text,
        score=score,
        infons=infons,
    )


def iter_bioc_passage_records(
    input_xml: PathLike | Sequence[PathLike],
    *,
    include_entities: bool = True,
) -> Iterator[PassageRecord]:
    """Yield passage records from one or more BioC XML files."""

    record_id = 0
    for path in _coerce_paths(input_xml):
        root = ET.parse(path).getroot()
        for document in root.findall(".//document"):
            document_id = _text_of(document, "id", "")
            for passage_id, passage in enumerate(document.findall("passage")):
                passage_offset = _int_text_of(passage, "offset", 0)
                text = _text_of(passage, "text", "")
                entities: list[EntitySpan] = []
                if include_entities:
                    entities = [
                        _annotation_to_entity_span(annotation, passage_offset)
                        for annotation in passage.findall("annotation")
                    ]
                yield PassageRecord(
                    record_id=record_id,
                    document_id=document_id,
                    passage_id=passage_id,
                    passage_offset=passage_offset,
                    text=text,
                    entities=entities,
                    infons=read_infons(passage),
                    source_path=str(path),
                )
                record_id += 1


def read_bioc_annotations(input_xml: PathLike | Sequence[PathLike]) -> list[PredictedEntity]:
    """Read all BioC annotations as entities with absolute offsets."""

    annotations: list[PredictedEntity] = []
    for record in iter_bioc_passage_records(input_xml, include_entities=True):
        for entity in record.entities:
            annotations.append(
                entity.to_absolute(
                    record.passage_offset,
                    document_id=record.document_id,
                    passage_id=record.passage_id,
                )
            )
    return annotations


def _remove_existing_annotations(root: ET.Element) -> None:
    for passage in root.findall(".//passage"):
        for annotation in list(passage.findall("annotation")):
            passage.remove(annotation)


def _group_predictions_by_passage(
    predicted_entities: Iterable[PredictedEntity],
) -> dict[tuple[str, int], list[PredictedEntity]]:
    grouped: dict[tuple[str, int], list[PredictedEntity]] = {}
    for entity in predicted_entities:
        key = (entity.document_id, int(entity.passage_id))
        grouped.setdefault(key, []).append(entity)
    return grouped


def _format_score(score: float) -> str:
    return f"{float(score):.6g}"


def _add_infon(parent: ET.Element, key: str, value: Any) -> None:
    ET.SubElement(parent, "infon", {"key": str(key)}).text = "" if value is None else str(value)


def write_predictions_to_bioc_xml(
    input_xml: PathLike,
    output_xml: PathLike,
    *,
    predicted_entities: Iterable[PredictedEntity],
    clear_existing: bool = True,
) -> Path:
    """Write predicted entities into a BioC XML file.

    ``PredictedEntity.start`` and ``PredictedEntity.end`` are interpreted as
    absolute BioC offsets. The written ``location`` element therefore uses the
    same absolute ``start`` value.
    """

    input_path = Path(input_xml)
    output_path = Path(output_xml)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(input_path)
    root = tree.getroot()

    if clear_existing:
        _remove_existing_annotations(root)

    grouped = _group_predictions_by_passage(predicted_entities)
    annotation_index = 0

    for document in root.findall(".//document"):
        document_id = _text_of(document, "id", "")
        passages = list(document.findall("passage"))
        for passage_id, passage in enumerate(passages):
            candidates = grouped.get((document_id, passage_id), [])
            # Also support predictions without a document id for simple one-file
            # examples; they are matched by passage index.
            candidates.extend(grouped.get(("", passage_id), []))

            for entity in candidates:
                annotation = ET.SubElement(passage, "annotation", {"id": f"T{annotation_index}"})
                annotation_index += 1

                _add_infon(annotation, "type", entity.label)
                for key, value in entity.infons.items():
                    if key == "type":
                        continue
                    _add_infon(annotation, key, value)
                if entity.score is not None:
                    _add_infon(annotation, "score", _format_score(entity.score))

                ET.SubElement(
                    annotation,
                    "location",
                    {
                        "offset": str(int(entity.start)),
                        "length": str(max(0, int(entity.end) - int(entity.start))),
                    },
                )
                ET.SubElement(annotation, "text").text = entity.text

    _indent_tree(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def collection_summary(input_xml: PathLike | Sequence[PathLike]) -> dict[str, int]:
    """Return simple counts for BioC XML files."""

    documents = 0
    passages = 0
    annotations = 0
    for path in _coerce_paths(input_xml):
        root = ET.parse(path).getroot()
        documents += len(root.findall(".//document"))
        passages += len(root.findall(".//passage"))
        annotations += len(root.findall(".//annotation"))
    return {"documents": documents, "passages": passages, "annotations": annotations}


def _record_to_json(record: PassageRecord, *, include_entities: bool) -> dict[str, Any]:
    row: dict[str, Any] = {
        "record_id": record.record_id,
        "document_id": record.document_id,
        "passage_id": record.passage_id,
        "passage_offset": record.passage_offset,
        "text": record.text,
    }
    if record.infons:
        row["infons"] = dict(record.infons)
    if include_entities:
        row["entities"] = [asdict(entity) for entity in record.entities]
    return row


def convert_bioc_to_jsonl(
    input_xml: PathLike | Sequence[PathLike],
    output_jsonl: PathLike,
    *,
    include_entities: bool = True,
) -> Path:
    """Convert BioC passages to JSON Lines."""

    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for record in iter_bioc_passage_records(input_xml, include_entities=include_entities):
            handle.write(json.dumps(_record_to_json(record, include_entities=include_entities), ensure_ascii=False))
            handle.write("\n")

    return output_path


def read_jsonl(path: PathLike) -> list[dict[str, Any]]:
    """Read a JSON Lines file."""

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


__all__ = [
    "EntitySpan",
    "PassageRecord",
    "PredictedEntity",
    "collection_summary",
    "convert_bioc_to_jsonl",
    "iter_bioc_passage_records",
    "read_bioc_annotations",
    "read_infons",
    "read_jsonl",
    "write_predictions_to_bioc_xml",
    "write_text_as_bioc_xml",
]
