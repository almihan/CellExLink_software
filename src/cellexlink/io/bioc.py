"""BioC XML utilities shared by CellExLink components.

This module uses Python's built-in :mod:`xml.etree.ElementTree` so lightweight
operations such as tests, examples, and normalization output parsing do not need
an additional BioC dependency. The recognition module can still use the external
``bioc`` package internally when convenient.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

from .jsonl import write_jsonl
from .schemas import BioCAnnotation, BioCPassage, EntitySpan, PassageRecord, PathLike, coerce_entity_span

LOGGER = logging.getLogger(__name__)

DEFAULT_EXCLUDED_TYPES = frozenset({"cell_vague"})


def iter_input_files(paths: Sequence[PathLike] | PathLike, *, suffixes: Optional[set[str]] = None) -> Iterator[Path]:
    """Yield files from a file, directory, or sequence of files/directories."""
    if isinstance(paths, (str, Path)):
        path_items: Sequence[PathLike] = [paths]
    else:
        path_items = paths

    normalized_suffixes = {suffix.lower() for suffix in suffixes} if suffixes else None

    for raw_path in path_items:
        path = Path(raw_path)
        if path.is_file():
            if normalized_suffixes is None or path.suffix.lower() in normalized_suffixes:
                yield path
        elif path.is_dir():
            for child in sorted(p for p in path.rglob("*") if p.is_file()):
                if normalized_suffixes is None or child.suffix.lower() in normalized_suffixes:
                    yield child
        else:
            raise FileNotFoundError(f"Input path does not exist: {path}")


def read_bioc_tree(path: PathLike) -> ET.ElementTree:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"BioC XML file does not exist: {path}")
    return ET.parse(path)


def write_bioc_tree(tree: ET.ElementTree, path: PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _indent(tree.getroot())
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def child_text(element: ET.Element, child_name: str, *, default: Optional[str] = None) -> Optional[str]:
    child = element.find(child_name)
    if child is None:
        return default
    return child.text if child.text is not None else default


def set_child_text(element: ET.Element, child_name: str, value: Any) -> ET.Element:
    child = element.find(child_name)
    if child is None:
        child = ET.SubElement(element, child_name)
    child.text = "" if value is None else str(value)
    return child


def get_infons(element: ET.Element) -> dict[str, str]:
    """Return BioC ``infon`` elements as a dictionary."""
    infons: dict[str, str] = {}
    for infon in element.findall("infon"):
        key = infon.attrib.get("key")
        if key:
            infons[key] = infon.text or ""
    return infons


def set_infon(element: ET.Element, key: str, value: Any) -> ET.Element:
    """Set or create a BioC ``infon`` under an element."""
    text = "" if value is None else str(value)
    for infon in element.findall("infon"):
        if infon.attrib.get("key") == key:
            infon.text = text
            return infon

    infon = ET.Element("infon", {"key": key})
    infon.text = text

    # Keep BioC-style order: infons before locations/text whenever possible.
    insert_index = 0
    for index, child in enumerate(list(element)):
        if child.tag == "infon":
            insert_index = index + 1
    element.insert(insert_index, infon)
    return infon


def remove_infons_by_prefix(element: ET.Element, prefix: str) -> int:
    """Remove infons whose key starts with ``prefix`` and return count."""
    removed = 0
    for infon in list(element.findall("infon")):
        if infon.attrib.get("key", "").startswith(prefix):
            element.remove(infon)
            removed += 1
    return removed


def iter_bioc_passages(path: PathLike) -> Iterator[BioCPassage]:
    """Yield BioC passages from one XML file."""
    tree = read_bioc_tree(path)
    root = tree.getroot()

    for document in root.findall(".//document"):
        document_id = child_text(document, "id", default="") or ""
        for passage_index, passage in enumerate(document.findall("passage")):
            passage_text = child_text(passage, "text", default="") or ""
            passage_offset = _safe_int(child_text(passage, "offset", default="0"), default=0)
            annotations = list(
                iter_bioc_annotations(
                    passage,
                    document_id=document_id,
                    passage_id=passage_index,
                    passage_offset=passage_offset,
                    passage_text=passage_text,
                )
            )
            yield BioCPassage(
                document_id=document_id,
                passage_id=passage_index,
                offset=passage_offset,
                text=passage_text,
                infons=get_infons(passage),
                annotations=annotations,
                source_path=str(path),
            )


def iter_bioc_annotations(
    passage_element: ET.Element,
    *,
    document_id: str,
    passage_id: int,
    passage_offset: int,
    passage_text: str,
) -> Iterator[BioCAnnotation]:
    """Yield annotations from one BioC passage.

    BioC offsets are normally absolute document offsets. The returned
    ``BioCAnnotation`` keeps absolute offsets, and ``as_entity_span()`` converts
    them to local passage offsets.
    """
    for annotation_index, annotation in enumerate(passage_element.findall("annotation")):
        ann_id = annotation.attrib.get("id", str(annotation_index))
        infons = get_infons(annotation)
        label = infons.get("type") or infons.get("entity_type") or infons.get("label") or "cell_type"
        annotation_text = child_text(annotation, "text", default="") or ""
        locations = annotation.findall("location")

        if not locations:
            LOGGER.warning("Skipping annotation %s because it has no BioC location.", ann_id)
            continue

        for location_index, location in enumerate(locations):
            absolute_start = _safe_int(location.attrib.get("offset"), default=-1)
            length = _safe_int(location.attrib.get("length"), default=-1)
            if absolute_start < 0 or length <= 0:
                LOGGER.warning(
                    "Skipping annotation %s with invalid location offset=%r length=%r.",
                    ann_id,
                    location.attrib.get("offset"),
                    location.attrib.get("length"),
                )
                continue

            absolute_end = absolute_start + length
            local_start = absolute_start - passage_offset
            local_end = local_start + length
            if 0 <= local_start <= local_end <= len(passage_text):
                actual_text = passage_text[local_start:local_end]
            else:
                actual_text = ""
                LOGGER.warning(
                    "Annotation %s location %d..%d is outside passage bounds offset=%d length=%d.",
                    ann_id,
                    absolute_start,
                    absolute_end,
                    passage_offset,
                    len(passage_text),
                )

            text = annotation_text or actual_text
            if annotation_text and actual_text and annotation_text != actual_text:
                LOGGER.warning(
                    "Annotation text mismatch for %s at %d..%d: XML text=%r, passage text=%r.",
                    ann_id,
                    absolute_start,
                    absolute_end,
                    annotation_text,
                    actual_text,
                )

            yield BioCAnnotation(
                ann_id=f"{ann_id}:{location_index}" if len(locations) > 1 else ann_id,
                document_id=document_id,
                passage_id=passage_id,
                passage_offset=passage_offset,
                start=absolute_start,
                end=absolute_end,
                text=text,
                label=label,
                infons=infons,
            )


def bioc_to_passage_records(
    srcs: Sequence[PathLike] | PathLike,
    *,
    include_entities: bool = True,
    exclude_types: Iterable[str] = DEFAULT_EXCLUDED_TYPES,
    skip_empty_passages: bool = True,
) -> Iterator[PassageRecord]:
    """Stream BioC XML files as CellExLink JSONL-compatible passage records."""
    excluded = {item.lower() for item in exclude_types}
    record_id = 0
    for src in iter_input_files(srcs, suffixes={".xml"}):
        for passage in iter_bioc_passages(src):
            if skip_empty_passages and not passage.text:
                continue
            record = passage.as_passage_record(include_entities=include_entities)
            record.record_id = record_id
            record.source_path = str(src)
            if include_entities and excluded:
                record.entities = [
                    entity for entity in record.entities if entity.label.lower() not in excluded
                ]
            yield record
            record_id += 1


def convert_bioc_to_jsonl(
    srcs: Sequence[PathLike] | PathLike,
    output_jsonl: PathLike,
    *,
    include_entities: bool = True,
    exclude_types: Iterable[str] = DEFAULT_EXCLUDED_TYPES,
    skip_empty_passages: bool = True,
) -> int:
    """Convert one or more BioC XML files into CellExLink JSONL records."""
    records = list(
        bioc_to_passage_records(
            srcs,
            include_entities=include_entities,
            exclude_types=exclude_types,
            skip_empty_passages=skip_empty_passages,
        )
    )
    write_jsonl(records, output_jsonl)
    return len(records)


def write_text_as_bioc(
    text: str,
    output_xml: PathLike,
    *,
    document_id: str = "doc0",
    passage_type: str = "text",
    source: str = "CellExLink",
    date: str = "",
    key: str = "cell_type_extraction",
) -> Path:
    """Write one plain-text string as a minimal BioC XML collection."""
    output_xml = Path(output_xml)
    output_xml.parent.mkdir(parents=True, exist_ok=True)

    xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<collection>
  <source>{escape(source)}</source>
  <date>{escape(date)}</date>
  <key>{escape(key)}</key>
  <document>
    <id>{escape(document_id)}</id>
    <passage>
      <infon key="type">{escape(passage_type)}</infon>
      <offset>0</offset>
      <text>{escape(text)}</text>
    </passage>
  </document>
</collection>
'''
    output_xml.write_text(xml, encoding="utf-8")
    return output_xml


def write_texts_as_bioc(
    texts: Mapping[str, str] | Sequence[str],
    output_xml: PathLike,
    *,
    passage_type: str = "text",
    source: str = "CellExLink",
    date: str = "",
    key: str = "cell_type_extraction",
) -> Path:
    """Write several documents as one minimal BioC XML collection."""
    if isinstance(texts, Mapping):
        items = list(texts.items())
    else:
        items = [(f"doc{index}", text) for index, text in enumerate(texts)]

    collection = ET.Element("collection")
    set_child_text(collection, "source", source)
    set_child_text(collection, "date", date)
    set_child_text(collection, "key", key)

    for document_id, text in items:
        document = ET.SubElement(collection, "document")
        set_child_text(document, "id", document_id)
        passage = ET.SubElement(document, "passage")
        set_infon(passage, "type", passage_type)
        set_child_text(passage, "offset", 0)
        set_child_text(passage, "text", text)

    return write_bioc_tree(ET.ElementTree(collection), output_xml)


def write_predictions_to_bioc(
    input_xml: PathLike,
    output_xml: PathLike,
    predictions: Mapping[tuple[str, int], Sequence[EntitySpan | Mapping[str, Any]]] | Sequence[PassageRecord | Mapping[str, Any]],
    *,
    clear_existing_annotations: bool = True,
    id_prefix: str = "T",
    label_infon_key: str = "type",
    source_infon: str = "CellExLink",
) -> Path:
    """Insert predicted entity spans into a copy of a BioC XML file.

    ``predictions`` can be either:

    1. A mapping from ``(document_id, passage_id)`` to entity spans.
    2. A sequence of :class:`PassageRecord` objects/dicts.
    """
    tree = read_bioc_tree(input_xml)
    lookup = _prediction_lookup(predictions)

    annotation_counter = 0
    for document in tree.getroot().findall(".//document"):
        document_id = child_text(document, "id", default="") or ""
        for passage_index, passage in enumerate(document.findall("passage")):
            passage_offset = _safe_int(child_text(passage, "offset", default="0"), default=0)
            passage_text = child_text(passage, "text", default="") or ""

            entities = list(lookup.get((document_id, passage_index), []))
            if not entities:
                continue

            if clear_existing_annotations:
                for annotation in list(passage.findall("annotation")):
                    passage.remove(annotation)

            for entity_value in entities:
                entity = coerce_entity_span(entity_value).with_text_from(passage_text)
                annotation_counter += 1
                annotation = ET.SubElement(passage, "annotation", {"id": f"{id_prefix}{annotation_counter}"})
                set_infon(annotation, label_infon_key, entity.label)
                set_infon(annotation, "source", source_infon)
                for key, value in entity.infons.items():
                    if value is not None:
                        set_infon(annotation, str(key), value)
                if entity.identifier:
                    set_infon(annotation, "identifier", entity.identifier)
                if entity.identifier_name:
                    set_infon(annotation, "identifier_name", entity.identifier_name)
                if entity.score is not None:
                    set_infon(annotation, "score", entity.score)
                if entity.source:
                    set_infon(annotation, "match_source", entity.source)

                ET.SubElement(
                    annotation,
                    "location",
                    {
                        "offset": str(entity.absolute_start(passage_offset)),
                        "length": str(entity.length),
                    },
                )
                set_child_text(annotation, "text", entity.text)

    return write_bioc_tree(tree, output_xml)


def clone_bioc_without_annotations(input_xml: PathLike, output_xml: PathLike) -> Path:
    """Copy a BioC XML file and remove all annotations from passages."""
    tree = read_bioc_tree(input_xml)
    for passage in tree.getroot().findall(".//passage"):
        for annotation in list(passage.findall("annotation")):
            passage.remove(annotation)
    return write_bioc_tree(tree, output_xml)


def collect_document_text(path: PathLike) -> dict[str, str]:
    """Collect all passage text per BioC document."""
    tree = read_bioc_tree(path)
    texts: dict[str, list[str]] = {}
    for document in tree.getroot().findall(".//document"):
        document_id = child_text(document, "id", default="") or ""
        texts.setdefault(document_id, [])
        for passage in document.findall("passage"):
            passage_text = child_text(passage, "text", default="") or ""
            if passage_text:
                texts[document_id].append(passage_text)
    return {document_id: "\n".join(parts) for document_id, parts in texts.items()}


def _prediction_lookup(
    predictions: Mapping[tuple[str, int], Sequence[EntitySpan | Mapping[str, Any]]] | Sequence[PassageRecord | Mapping[str, Any]],
) -> dict[tuple[str, int], list[EntitySpan | Mapping[str, Any]]]:
    if isinstance(predictions, Mapping):
        return {key: list(value) for key, value in predictions.items()}

    lookup: dict[tuple[str, int], list[EntitySpan | Mapping[str, Any]]] = {}
    for record_value in predictions:
        if isinstance(record_value, PassageRecord):
            record = record_value
        else:
            record = PassageRecord.from_dict(record_value)
        lookup[(record.document_id, record.passage_id)] = list(record.entities)
    return lookup


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _indent(element: ET.Element, level: int = 0) -> None:
    """Pretty-print XML in-place for stable, reviewable outputs."""
    indent_text = "\n" + level * "  "
    children = list(element)
    if children:
        if not element.text or not element.text.strip():
            element.text = indent_text + "  "
        for child in children:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():  # type: ignore[possibly-undefined]
            child.tail = indent_text  # type: ignore[possibly-undefined]
    if level and (not element.tail or not element.tail.strip()):
        element.tail = indent_text


def copy_element(element: ET.Element) -> ET.Element:
    """Return a deep copy of an ElementTree element."""
    return copy.deepcopy(element)
