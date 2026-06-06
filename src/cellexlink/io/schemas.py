"""Shared data schemas for CellExLink input/output records.

They are used by the recognition, normalization, examples, tests, and benchmark
code to avoid each component inventing a slightly different JSON/BioC schema.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

PathLike = str | Path


@dataclass(slots=True)
class EntitySpan:
    """A flat entity span inside one passage.

    Offsets are local to the passage by default. Convert to BioC absolute
    offsets with :meth:`absolute_start` and :meth:`absolute_end`.
    """

    start: int
    end: int
    label: str = "cell_type"
    text: str = ""
    ann_id: str = ""
    identifier: Optional[str] = None
    identifier_name: Optional[str] = None
    score: Optional[float] = None
    source: Optional[str] = None
    infons: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.start = int(self.start)
        self.end = int(self.end)
        if self.start < 0:
            raise ValueError(f"EntitySpan.start must be non-negative, got {self.start}.")
        if self.end <= self.start:
            raise ValueError(f"EntitySpan.end must be greater than start, got {self.start}..{self.end}.")
        if not self.label:
            self.label = "cell_type"

    @property
    def length(self) -> int:
        return self.end - self.start

    def absolute_start(self, passage_offset: int) -> int:
        return int(passage_offset) + self.start

    def absolute_end(self, passage_offset: int) -> int:
        return int(passage_offset) + self.end

    def with_text_from(self, passage_text: str) -> "EntitySpan":
        """Return a copy whose text is filled from the passage when missing."""
        if self.text:
            return self
        return EntitySpan(
            start=self.start,
            end=self.end,
            label=self.label,
            text=passage_text[self.start : self.end],
            ann_id=self.ann_id,
            identifier=self.identifier,
            identifier_name=self.identifier_name,
            score=self.score,
            source=self.source,
            infons=dict(self.infons),
        )

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EntitySpan":
        start = data.get("start")
        end = data.get("end")
        if start is None or end is None:
            raise ValueError("EntitySpan requires 'start' and 'end'.")
        return cls(
            start=int(start),
            end=int(end),
            label=str(data.get("label") or data.get("type") or "cell_type"),
            text=str(data.get("text") or ""),
            ann_id=str(data.get("ann_id") or data.get("id") or ""),
            identifier=_optional_str(data.get("identifier") or data.get("cl_id")),
            identifier_name=_optional_str(data.get("identifier_name") or data.get("cl_label")),
            score=_optional_float(data.get("score")),
            source=_optional_str(data.get("source")),
            infons=dict(data.get("infons") or {}),
        )


@dataclass(slots=True)
class PassageRecord:
    """One text passage and its optional local-offset entity annotations."""

    record_id: int | str
    document_id: str
    passage_id: int
    passage_offset: int
    text: str
    entities: list[EntitySpan] = field(default_factory=list)
    infons: dict[str, Any] = field(default_factory=dict)
    source_path: Optional[str] = None

    def __post_init__(self) -> None:
        self.passage_id = int(self.passage_id)
        self.passage_offset = int(self.passage_offset or 0)
        self.text = self.text or ""
        self.entities = [coerce_entity_span(entity) for entity in self.entities]

        for entity in self.entities:
            if entity.end > len(self.text):
                raise ValueError(
                    "Entity span exceeds passage length: "
                    f"{entity.start}..{entity.end} for passage length {len(self.text)}."
                )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["entities"] = [entity.to_dict() for entity in self.entities]
        return _drop_none(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PassageRecord":
        return cls(
            record_id=data.get("record_id", data.get("id", "")),
            document_id=str(data.get("document_id") or data.get("doc_id") or ""),
            passage_id=int(data.get("passage_id", data.get("passage_index", 0)) or 0),
            passage_offset=int(data.get("passage_offset", data.get("offset", 0)) or 0),
            text=str(data.get("text") or ""),
            entities=[coerce_entity_span(entity) for entity in data.get("entities", [])],
            infons=dict(data.get("infons") or {}),
            source_path=_optional_str(data.get("source_path")),
        )


@dataclass(slots=True)
class BioCAnnotation:
    """A BioC annotation converted into a simple Python object."""

    ann_id: str
    document_id: str
    passage_id: int
    passage_offset: int
    start: int
    end: int
    text: str
    label: str = "cell_type"
    infons: dict[str, str] = field(default_factory=dict)

    @property
    def local_start(self) -> int:
        return self.start - self.passage_offset

    @property
    def local_end(self) -> int:
        return self.end - self.passage_offset

    def as_entity_span(self) -> EntitySpan:
        return EntitySpan(
            start=self.local_start,
            end=self.local_end,
            label=self.label,
            text=self.text,
            ann_id=self.ann_id,
            infons=dict(self.infons),
        )

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(asdict(self))


@dataclass(slots=True)
class BioCPassage:
    """A BioC passage plus parsed annotations."""

    document_id: str
    passage_id: int
    offset: int
    text: str
    infons: dict[str, str] = field(default_factory=dict)
    annotations: list[BioCAnnotation] = field(default_factory=list)
    source_path: Optional[str] = None

    def as_passage_record(self, *, include_entities: bool = True) -> PassageRecord:
        entities = [ann.as_entity_span() for ann in self.annotations] if include_entities else []
        return PassageRecord(
            record_id=f"{self.document_id}:{self.passage_id}",
            document_id=self.document_id,
            passage_id=self.passage_id,
            passage_offset=self.offset,
            text=self.text,
            entities=entities,
            infons=dict(self.infons),
            source_path=self.source_path,
        )

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(asdict(self))


@dataclass(slots=True)
class LinkedEntity:
    """A normalized cell-type mention."""

    mention: str
    start: Optional[int] = None
    end: Optional[int] = None
    cl_id: Optional[str] = None
    cl_label: Optional[str] = None
    score: Optional[float] = None
    source: Optional[str] = None
    document_id: str = ""
    passage_id: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(asdict(self))


@dataclass(slots=True)
class CellExLinkDocumentResult:
    """Container for all extracted/normalized entities in one document."""

    document_id: str
    entities: list[LinkedEntity] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["entities"] = [entity.to_dict() for entity in self.entities]
        return _drop_none(data)


def coerce_entity_span(value: EntitySpan | Mapping[str, Any]) -> EntitySpan:
    if isinstance(value, EntitySpan):
        return value
    if isinstance(value, Mapping):
        return EntitySpan.from_dict(value)
    raise TypeError(f"Expected EntitySpan or mapping, got {type(value)!r}.")


def coerce_passage_record(value: PassageRecord | Mapping[str, Any]) -> PassageRecord:
    if isinstance(value, PassageRecord):
        return value
    if isinstance(value, Mapping):
        return PassageRecord.from_dict(value)
    raise TypeError(f"Expected PassageRecord or mapping, got {type(value)!r}.")


def spans_overlap(left: EntitySpan, right: EntitySpan) -> bool:
    return max(left.start, right.start) < min(left.end, right.end)


def find_overlapping_spans(entities: Sequence[EntitySpan]) -> list[tuple[EntitySpan, EntitySpan]]:
    overlaps: list[tuple[EntitySpan, EntitySpan]] = []
    for index, left in enumerate(entities):
        for right in entities[index + 1 :]:
            if spans_overlap(left, right):
                overlaps.append((left, right))
    return overlaps


def ensure_no_overlapping_spans(entities: Sequence[EntitySpan]) -> None:
    overlaps = find_overlapping_spans(entities)
    if overlaps:
        left, right = overlaps[0]
        raise ValueError(
            "Overlapping entity spans are not supported by this operation: "
            f"{left.start}..{left.end} overlaps {right.start}..{right.end}."
        )


def to_jsonable(value: Any) -> Any:
    """Convert dataclasses and paths into JSON-serializable objects."""
    if is_dataclass(value):
        if hasattr(value, "to_dict"):
            return value.to_dict()
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def _drop_none(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if value is not None}


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
