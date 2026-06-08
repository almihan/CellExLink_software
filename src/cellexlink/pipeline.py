"""Public Python API for CellExLink.

The :class:`CellExLinkPipeline` class is intentionally the only high-level
object users need for the SoftwareX release. It exposes the three supported
workflows:

1. NER only: detect cell-type mentions.
2. NEN only: normalize existing mentions to Cell Ontology identifiers.
3. End-to-end extraction: run NER followed by NEN.

Heavy model imports are kept inside methods so ``import cellexlink`` remains
lightweight and works without loading Bioformer or SapBERT checkpoints.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping
from xml.etree import ElementTree as ET

PathLike = str | Path

DEFAULT_NER_MODEL = "almire/CellExLink-bioformer16L"
DEFAULT_NEN_MODEL = "almire/CellExLink-Sapbert"


def _compact_dict(obj: Any) -> dict[str, Any]:
    """Convert a dataclass object to a compact public dictionary.

    Empty metadata fields such as ``infons={}`` and ``candidates=[]`` are
    omitted from JSON/JSONL output. ``None`` values are also omitted so the
    public API output remains easy to read.
    """

    data = asdict(obj)
    return {
        key: value
        for key, value in data.items()
        if value is not None and value != {} and value != []
    }


@dataclass(slots=True)
class RecognizedMention:
    """One NER-only cell-type mention prediction."""

    document_id: str
    passage_index: int
    mention: str
    start: int | None = None
    end: int | None = None
    entity_type: str | None = "cell_type"
    score: float | None = None
    infons: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _compact_dict(self)


@dataclass(slots=True)
class NormalizedMention:
    """One NEN-only Cell Ontology normalization prediction."""

    mention: str
    cl_id: str | None = None
    cl_label: str | None = None
    score: float | None = None
    source: str | None = None
    normalized_text: str | None = None
    document_id: str | None = None
    start: int | None = None
    end: int | None = None
    entity_type: str | None = None
    candidates: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _compact_dict(self)


@dataclass(slots=True)
class ExtractionResult:
    """One end-to-end CellExLink prediction.

    The result contains both the detected text span and, when available, the
    linked Cell Ontology identifier.
    """

    document_id: str
    passage_index: int
    mention: str
    start: int | None = None
    end: int | None = None
    entity_type: str | None = None
    cl_id: str | None = None
    cl_label: str | None = None
    score: float | None = None
    source: str | None = None
    infons: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _compact_dict(self)


MentionInput = str | RecognizedMention | ExtractionResult | Mapping[str, Any]


@dataclass(slots=True)
class _MentionMeta:
    mention: str
    document_id: str | None = None
    start: int | None = None
    end: int | None = None
    entity_type: str | None = None


@dataclass(slots=True)
class CellExLinkPipeline:
    """High-level CellExLink pipeline.

    Examples
    --------
    >>> from cellexlink import CellExLinkPipeline
    >>> pipe = CellExLinkPipeline.from_pretrained(
    ...     ner_model="models/CellExLink-bioformer16L",
    ...     nen_model="models/CellExLink-Sapbert",
    ... )
    >>> pipe.recognize_text("CD8+ T cells were detected.")  # NER only
    >>> pipe.normalize_mentions(["CD8+ T cells"])          # NEN only
    >>> pipe.extract_text("CD8+ T cells were detected.")   # End-to-end
    """

    ner_model: PathLike = DEFAULT_NER_MODEL
    nen_model: PathLike = DEFAULT_NEN_MODEL
    ontology_path: PathLike | None = None
    abbreviations_path: PathLike | None = None
    disable_abbreviations: bool = False
    output_dir: PathLike = "cellexlink_outputs"
    warmup_runs: int = 1
    batch_size: int = 16
    fp16: bool = False
    trust_remote_code: bool = False

    @classmethod
    def from_pretrained(
        cls,
        ner_model: PathLike = DEFAULT_NER_MODEL,
        nen_model: PathLike = DEFAULT_NEN_MODEL,
        **kwargs: Any,
    ) -> "CellExLinkPipeline":
        """Create a pipeline from local paths or Hugging Face model IDs."""

        return cls(ner_model=ner_model, nen_model=nen_model, **kwargs)

    # ------------------------------------------------------------------
    # BioC XML API
    # ------------------------------------------------------------------
    def recognize_bioc(
        self,
        input_xml: PathLike,
        output_xml: PathLike,
        *,
        output_dir: PathLike | None = None,
        overwrite: bool = True,
    ) -> Path:
        """Run NER only on a BioC XML file.

        Parameters
        ----------
        input_xml:
            BioC XML input file.
        output_xml:
            BioC XML output file containing predicted cell-type annotations.
        output_dir:
            Optional working directory for the lower-level recognition code.
        overwrite:
            If ``False``, fail when ``output_xml`` already exists.
        """

        from cellexlink.recognition.predict import predict_ner

        input_xml = Path(input_xml)
        output_xml = Path(output_xml)
        if not input_xml.is_file():
            raise FileNotFoundError(f"Input BioC XML file does not exist: {input_xml}")
        if output_xml.exists() and not overwrite:
            raise FileExistsError(f"Output file already exists: {output_xml}")

        run_dir = Path(output_dir or Path(self.output_dir) / "ner")
        run_dir.mkdir(parents=True, exist_ok=True)
        output_xml.parent.mkdir(parents=True, exist_ok=True)

        result = predict_ner(
            model_path=self.ner_model,
            input_xml=input_xml,
            output_dir=run_dir,
            output_xml=output_xml,
            warmup_runs=self.warmup_runs,
            per_device_predict_batch_size=self.batch_size,
            fp16=self.fp16,
            trust_remote_code=self.trust_remote_code,
        )

        if isinstance(result, int) and result != 0:
            raise RuntimeError(
                f"CellExLink NER failed with exit code {result}. "
                f"Check the logs in: {run_dir}"
            )
        return output_xml

    def normalize_bioc(
        self,
        input_xml: PathLike,
        output_xml: PathLike,
        *,
        overwrite: bool = True,
    ) -> Path:
        """Run NEN only on a BioC XML file that already has mention spans.

        This method is used for gold-span NEN evaluation and for normalizing
        annotations produced by any compatible NER system. The normalizer does
        not need gold CL IDs; if gold ``identifier`` infons are present, they
        are preserved for later evaluation.
        """

        from cellexlink.normalization.linker import normalize_bioc

        input_xml = Path(input_xml)
        output_xml = Path(output_xml)
        if not input_xml.is_file():
            raise FileNotFoundError(f"Input BioC XML file does not exist: {input_xml}")
        if output_xml.exists() and not overwrite:
            raise FileExistsError(f"Output file already exists: {output_xml}")
        output_xml.parent.mkdir(parents=True, exist_ok=True)

        kwargs: dict[str, Any] = {
            "input_xml": input_xml,
            "output_xml": output_xml,
            "model_path": self.nen_model,
            "disable_abbreviations": self.disable_abbreviations,
            "batch_size": self.batch_size,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.ontology_path is not None:
            # ``cell_types`` is kept for compatibility with the original scripts.
            kwargs["cell_types"] = self.ontology_path
        if self.abbreviations_path is not None:
            kwargs["abbreviations"] = self.abbreviations_path

        normalize_bioc(**kwargs)
        return output_xml

    def extract_bioc(
        self,
        input_xml: PathLike,
        output_xml: PathLike,
        *,
        ner_output_xml: PathLike | None = None,
        output_dir: PathLike | None = None,
        overwrite: bool = True,
    ) -> Path:
        """Run end-to-end NER + NEN on a BioC XML file."""

        input_xml = Path(input_xml)
        output_xml = Path(output_xml)
        if not input_xml.is_file():
            raise FileNotFoundError(f"Input BioC XML file does not exist: {input_xml}")
        if output_xml.exists() and not overwrite:
            raise FileExistsError(f"Output file already exists: {output_xml}")

        run_dir = Path(output_dir or self.output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        if ner_output_xml is None:
            ner_output_xml = run_dir / "ner_predictions.xml"
        ner_output_xml = Path(ner_output_xml)
        ner_output_xml.parent.mkdir(parents=True, exist_ok=True)
        output_xml.parent.mkdir(parents=True, exist_ok=True)

        self.recognize_bioc(
            input_xml=input_xml,
            output_xml=ner_output_xml,
            output_dir=run_dir / "ner",
            overwrite=True,
        )
        self.normalize_bioc(
            input_xml=ner_output_xml,
            output_xml=output_xml,
            overwrite=True,
        )
        return output_xml

    # ------------------------------------------------------------------
    # Plain-text API
    # ------------------------------------------------------------------
    def recognize_text(
        self,
        text: str,
        *,
        document_id: str = "doc0",
        output_dir: PathLike | None = None,
    ) -> list[RecognizedMention]:
        """Run NER only on one plain-text string."""

        self._validate_text(text)
        if not text.strip():
            return []

        from cellexlink.io import write_text_as_bioc_xml

        run_dir = Path(output_dir or self.output_dir) / "text_ner"
        run_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="cellexlink_ner_text_") as tmp:
            tmp_dir = Path(tmp)
            input_xml = tmp_dir / "input.xml"
            ner_xml = tmp_dir / "ner.xml"
            write_text_as_bioc_xml(text, input_xml, document_id=document_id)
            self.recognize_bioc(input_xml, ner_xml, output_dir=run_dir)
            return self.read_recognized_mentions_from_bioc(ner_xml)

    def normalize_mentions(
        self,
        mentions: Iterable[MentionInput],
        *,
        document_text: str | None = None,
        document_id: str = "doc0",
        topn: int = 1,
    ) -> list[NormalizedMention]:
        """Run NEN only on existing mention strings.

        Parameters
        ----------
        mentions:
            Mention strings, dictionaries, :class:`RecognizedMention` objects,
            or :class:`ExtractionResult` objects. Dictionaries should contain a
            ``text`` or ``mention`` field and can optionally contain ``start``,
            ``end``, and ``entity_type``.
        document_text:
            Optional source document text. Supplying this enables document-level
            Ab3P abbreviation recovery for ambiguous short forms.
        document_id:
            Key used for the optional document text.
        topn:
            Number of ranked CL candidates to retain per mention.
        """

        from cellexlink.normalization.linker import (
            MentionRecord,
            normalize_mentions as link_mentions,
        )

        mention_meta = [_coerce_mention_input(item, default_document_id=document_id) for item in mentions]
        if not mention_meta:
            return []

        doc_text_by_key = {document_id: document_text} if document_text else None
        records = [
            MentionRecord(
                mention_text=item.mention,
                document_key=document_id if document_text else item.document_id,
            )
            for item in mention_meta
        ]

        results = link_mentions(
            records,
            ontology_path=self.ontology_path,
            model_path=self.nen_model,
            abbreviations_path=self.abbreviations_path,
            disable_abbreviations=self.disable_abbreviations,
            document_text_by_key=doc_text_by_key,
            batch_size=self.batch_size,
            topn=topn,
            trust_remote_code=self.trust_remote_code,
        )
        return [
            self._normalized_mention_from_result(result, meta)
            for result, meta in zip(results, mention_meta, strict=False)
        ]

    def extract_text(
        self,
        text: str,
        *,
        document_id: str = "doc0",
        output_dir: PathLike | None = None,
    ) -> list[ExtractionResult]:
        """Run end-to-end NER + NEN on one plain-text string."""

        self._validate_text(text)
        if not text.strip():
            return []

        from cellexlink.io import write_text_as_bioc_xml

        run_dir = Path(output_dir or self.output_dir) / "text_end_to_end"
        run_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="cellexlink_e2e_text_") as tmp:
            tmp_dir = Path(tmp)
            input_xml = tmp_dir / "input.xml"
            ner_xml = tmp_dir / "ner.xml"
            normalized_xml = tmp_dir / "normalized.xml"
            write_text_as_bioc_xml(text, input_xml, document_id=document_id)
            self.extract_bioc(
                input_xml=input_xml,
                output_xml=normalized_xml,
                ner_output_xml=ner_xml,
                output_dir=run_dir,
            )
            return self.read_predictions_from_bioc(normalized_xml)

    def extract_text_file(
        self,
        input_txt: PathLike,
        output_jsonl: PathLike,
        *,
        document_id: str = "doc0",
        output_dir: PathLike | None = None,
    ) -> Path:
        """Run end-to-end extraction on a plain-text file and write JSONL."""

        input_txt = Path(input_txt)
        if not input_txt.is_file():
            raise FileNotFoundError(f"Input text file does not exist: {input_txt}")
        predictions = self.extract_text(
            input_txt.read_text(encoding="utf-8"),
            document_id=document_id,
            output_dir=output_dir,
        )
        return write_predictions_jsonl(predictions, output_jsonl)

    # ------------------------------------------------------------------
    # Readers / writers
    # ------------------------------------------------------------------
    @staticmethod
    def read_recognized_mentions_from_bioc(xml_path: PathLike) -> list[RecognizedMention]:
        """Read NER-only predictions from BioC XML."""

        from cellexlink.io import read_bioc_annotations

        mentions: list[RecognizedMention] = []
        for entity in read_bioc_annotations(xml_path):
            mentions.append(
                RecognizedMention(
                    document_id=entity.document_id,
                    passage_index=entity.passage_id,
                    mention=entity.text,
                    start=entity.start,
                    end=entity.end,
                    entity_type=entity.label,
                    score=entity.score,
                    infons=dict(entity.infons),
                )
            )
        return mentions

    @staticmethod
    def read_predictions_from_bioc(xml_path: PathLike) -> list[ExtractionResult]:
        """Read end-to-end predictions from normalized BioC XML."""

        xml_path = Path(xml_path)
        if not xml_path.is_file():
            raise FileNotFoundError(f"BioC XML file does not exist: {xml_path}")

        tree = ET.parse(xml_path)
        root = tree.getroot()
        results: list[ExtractionResult] = []

        for document in root.findall(".//document"):
            doc_id_node = document.find("id")
            document_id = doc_id_node.text if doc_id_node is not None and doc_id_node.text else ""

            for passage_index, passage in enumerate(document.findall("passage")):
                for annotation in passage.findall("annotation"):
                    mention = _find_child_text(annotation, "text") or ""
                    start: int | None = None
                    end: int | None = None
                    location = annotation.find("location")
                    if location is not None:
                        start = _safe_int(location.attrib.get("offset"), default=None)
                        length = _safe_int(location.attrib.get("length"), default=None)
                        if start is not None and length is not None:
                            end = start + length

                    infons = _read_infons(annotation)
                    entity_type = infons.get("type") or infons.get("entity_type") or infons.get("label")
                    cl_id = _find_first_infon_value_by_suffix(infons, "_id_0")
                    cl_label = _find_first_infon_value_by_suffix(infons, "_identifier_name_0")
                    score = _safe_float(_find_first_infon_value_by_suffix(infons, "_identifier_score_0"))
                    source = _find_first_infon_value_by_suffix(infons, "_match_source")

                    results.append(
                        ExtractionResult(
                            document_id=document_id,
                            passage_index=passage_index,
                            mention=mention,
                            start=start,
                            end=end,
                            entity_type=entity_type,
                            cl_id=cl_id,
                            cl_label=cl_label,
                            score=score,
                            source=source,
                            infons=infons,
                        )
                    )
        return results

    @staticmethod
    def _normalized_mention_from_result(result: Any, meta: _MentionMeta | None = None) -> NormalizedMention:
        best = result.best
        if best is None:
            return NormalizedMention(
                mention=result.mention_text,
                normalized_text=getattr(result, "normalized_text", None),
                document_id=getattr(result, "document_key", None) or (meta.document_id if meta else None),
                start=meta.start if meta else None,
                end=meta.end if meta else None,
                entity_type=meta.entity_type if meta else None,
                candidates=[],
            )
        return NormalizedMention(
            mention=result.mention_text,
            cl_id=best.identifier,
            cl_label=best.preferred_label or best.name,
            score=best.final_score,
            source=best.source,
            normalized_text=getattr(result, "normalized_text", None),
            document_id=getattr(result, "document_key", None) or (meta.document_id if meta else None),
            start=meta.start if meta else None,
            end=meta.end if meta else None,
            entity_type=meta.entity_type if meta else None,
            candidates=[candidate.to_dict() for candidate in result.candidates],
        )

    @staticmethod
    def _validate_text(text: str) -> None:
        if not isinstance(text, str):
            raise TypeError("text must be a string")


# ----------------------------------------------------------------------
# Mention coercion helpers
# ----------------------------------------------------------------------
def _coerce_mention_input(item: MentionInput, *, default_document_id: str) -> _MentionMeta:
    if isinstance(item, RecognizedMention):
        return _MentionMeta(
            mention=item.mention,
            document_id=item.document_id or default_document_id,
            start=item.start,
            end=item.end,
            entity_type=item.entity_type,
        )

    if isinstance(item, ExtractionResult):
        return _MentionMeta(
            mention=item.mention,
            document_id=item.document_id or default_document_id,
            start=item.start,
            end=item.end,
            entity_type=item.entity_type,
        )

    if isinstance(item, Mapping):
        mention = item.get("text") or item.get("mention") or item.get("mention_text")
        if mention is None:
            raise ValueError("Mention dictionaries must contain 'text', 'mention', or 'mention_text'.")
        start = _safe_int(item.get("start"), default=None)
        end = _safe_int(item.get("end"), default=None)
        if end is None and start is not None:
            length = _safe_int(item.get("length"), default=None)
            if length is not None:
                end = start + length
        return _MentionMeta(
            mention=str(mention),
            document_id=str(item.get("document_id") or default_document_id),
            start=start,
            end=end,
            entity_type=(str(item.get("entity_type")) if item.get("entity_type") is not None else None),
        )

    return _MentionMeta(mention=str(item), document_id=default_document_id)


# ----------------------------------------------------------------------
# Small XML/JSON helpers
# ----------------------------------------------------------------------
def _read_infons(element: ET.Element) -> dict[str, str]:
    infons: dict[str, str] = {}
    for infon in element.findall("infon"):
        key = infon.attrib.get("key")
        if key:
            infons[key] = infon.text or ""
    return infons


def _find_child_text(element: ET.Element, child_name: str) -> str | None:
    child = element.find(child_name)
    if child is None:
        return None
    return child.text


def _find_first_infon_value_by_suffix(infons: dict[str, str], suffix: str) -> str | None:
    for key, value in infons.items():
        if key.endswith(suffix):
            return value
    return None


def _safe_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _jsonable(item: Any) -> dict[str, Any]:
    if hasattr(item, "to_dict"):
        return item.to_dict()
    if hasattr(item, "__dataclass_fields__"):
        return _compact_dict(item)
    if isinstance(item, dict):
        return {
            key: value
            for key, value in item.items()
            if value is not None and value != {} and value != []
        }
    raise TypeError(f"Object is not JSON serializable by CellExLink: {type(item)!r}")


def write_predictions_jsonl(predictions: Iterable[Any], output_path: PathLike) -> Path:
    """Write CellExLink API results to JSON Lines."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(_jsonable(prediction), ensure_ascii=False) + "\n")
    return output_path


__all__ = [
    "DEFAULT_NEN_MODEL",
    "DEFAULT_NER_MODEL",
    "CellExLinkPipeline",
    "ExtractionResult",
    "MentionInput",
    "NormalizedMention",
    "RecognizedMention",
    "write_predictions_jsonl",
]
