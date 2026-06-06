from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape


PathLike = str | Path


DEFAULT_NER_MODEL = "almire/CellExLink-bioformer16L"
DEFAULT_NEN_MODEL = "almire/CellExLink-Sapbert"


@dataclass(slots=True)
class ExtractionResult:
    """
    One CellExLink prediction.

    Attributes
    ----------
    document_id:
        BioC document identifier.
    passage_index:
        Index of the passage inside the BioC document.
    mention:
        Mention text detected by the NER model.
    start:
        Character start offset.
    end:
        Character end offset.
    entity_type:
        Entity type from the BioC annotation, usually a cell/cell-type label.
    cl_id:
        Predicted Cell Ontology identifier, for example "CL:0000077".
    cl_label:
        Preferred label or matched ontology name when available.
    score:
        Normalization score when available.
    source:
        Normalization source, for example model match or abbreviation match.
    """

    document_id: str
    passage_index: int
    mention: str
    start: Optional[int] = None
    end: Optional[int] = None
    entity_type: Optional[str] = None
    cl_id: Optional[str] = None
    cl_label: Optional[str] = None
    score: Optional[float] = None
    source: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CellExLinkPipeline:
    """
    End-to-end CellExLink pipeline.

    This class is the main public Python API. It runs:

    1. Cell-type recognition.
    2. Cell Ontology normalization.

    The heavy model loading is intentionally kept inside the recognition and
    normalization modules, so importing cellexlink remains lightweight.
    """

    ner_model: PathLike = DEFAULT_NER_MODEL
    nen_model: PathLike = DEFAULT_NEN_MODEL
    ontology_path: Optional[PathLike] = None
    abbreviations_path: Optional[PathLike] = None
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
        """
        Create a pipeline using pretrained CellExLink checkpoints.

        Examples
        --------
        >>> from cellexlink import CellExLinkPipeline
        >>> pipe = CellExLinkPipeline.from_pretrained()
        """
        return cls(
            ner_model=ner_model,
            nen_model=nen_model,
            **kwargs,
        )

    def extract_bioc(
        self,
        input_xml: PathLike,
        output_xml: PathLike,
        *,
        ner_output_xml: Optional[PathLike] = None,
        output_dir: Optional[PathLike] = None,
        overwrite: bool = True,
    ) -> Path:
        """
        Run the full end-to-end CellExLink pipeline on a BioC XML file.

        Parameters
        ----------
        input_xml:
            Input BioC XML file.
        output_xml:
            Final normalized BioC XML output file.
        ner_output_xml:
            Optional path for intermediate NER-only BioC XML output.
        output_dir:
            Directory for intermediate model outputs.
        overwrite:
            Whether to overwrite existing output files.

        Returns
        -------
        Path
            Path to the final normalized BioC XML file.
        """
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
        else:
            ner_output_xml = Path(ner_output_xml)

        ner_output_xml = Path(ner_output_xml)
        ner_output_xml.parent.mkdir(parents=True, exist_ok=True)
        output_xml.parent.mkdir(parents=True, exist_ok=True)

        self.recognize_bioc(
            input_xml=input_xml,
            output_xml=ner_output_xml,
            output_dir=run_dir / "ner",
        )

        self.normalize_bioc(
            input_xml=ner_output_xml,
            output_xml=output_xml,
        )

        return output_xml

    def recognize_bioc(
        self,
        input_xml: PathLike,
        output_xml: PathLike,
        *,
        output_dir: Optional[PathLike] = None,
    ) -> Path:
        """
        Run only the CellExLink NER component on BioC XML.

        This function expects the lower-level wrapper:

            cellexlink.recognition.predict.predict_ner

        to exist after you move/refactor the old recognition code.
        """
        from cellexlink.recognition.predict import predict_ner

        input_xml = Path(input_xml)
        output_xml = Path(output_xml)
        run_dir = Path(output_dir or Path(self.output_dir) / "ner")

        if not input_xml.is_file():
            raise FileNotFoundError(f"Input BioC XML file does not exist: {input_xml}")

        run_dir.mkdir(parents=True, exist_ok=True)
        output_xml.parent.mkdir(parents=True, exist_ok=True)

        return_code = predict_ner(
            model_path=self.ner_model,
            input_xml=input_xml,
            output_dir=run_dir,
            output_xml=output_xml,
            warmup_runs=self.warmup_runs,
            per_device_predict_batch_size=self.batch_size,
            fp16=self.fp16,
            trust_remote_code=self.trust_remote_code,
        )

        if return_code != 0:
            raise RuntimeError(
                f"CellExLink NER failed with exit code {return_code}. "
                f"Check the logs in: {run_dir}"
            )

        return output_xml

    def normalize_bioc(
        self,
        input_xml: PathLike,
        output_xml: PathLike,
    ) -> Path:
        """
        Run only the CellExLink normalization component on BioC XML.

        The input XML should already contain cell-type annotations, either gold
        mentions or mentions predicted by the CellExLink NER model.

        This function expects the lower-level wrapper:

            cellexlink.normalization.linker.normalize_bioc

        to exist after you move/refactor the old normalization code.
        """
        from cellexlink.normalization.linker import normalize_bioc

        input_xml = Path(input_xml)
        output_xml = Path(output_xml)

        if not input_xml.is_file():
            raise FileNotFoundError(f"Input BioC XML file does not exist: {input_xml}")

        output_xml.parent.mkdir(parents=True, exist_ok=True)

        kwargs: dict[str, Any] = {
            "input_xml": input_xml,
            "output_xml": output_xml,
            "model_path": self.nen_model,
            "disable_abbreviations": self.disable_abbreviations,
        }

        if self.ontology_path is not None:
            kwargs["cell_types"] = self.ontology_path

        if self.abbreviations_path is not None:
            kwargs["abbreviations"] = self.abbreviations_path

        normalize_bioc(**kwargs)

        return output_xml

    def extract_text(
        self,
        text: str,
        *,
        document_id: str = "doc0",
        output_dir: Optional[PathLike] = None,
    ) -> list[ExtractionResult]:
        """
        Run end-to-end CellExLink on a plain text string.

        Internally, the text is converted to a minimal BioC XML document,
        then the regular BioC pipeline is used.

        This keeps one canonical pipeline for both BioC and plain text input.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text.strip():
            return []

        run_dir = Path(output_dir or self.output_dir) / "text_prediction"
        run_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="cellexlink_text_") as tmp:
            tmp_dir = Path(tmp)
            input_xml = tmp_dir / "input.xml"
            ner_xml = tmp_dir / "ner_predictions.xml"
            normalized_xml = tmp_dir / "normalized.xml"

            self._write_text_as_bioc_xml(
                text=text,
                output_xml=input_xml,
                document_id=document_id,
            )

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
        output_dir: Optional[PathLike] = None,
    ) -> Path:
        """
        Run CellExLink on a plain text file and write JSONL predictions.
        """
        input_txt = Path(input_txt)
        output_jsonl = Path(output_jsonl)

        if not input_txt.is_file():
            raise FileNotFoundError(f"Input text file does not exist: {input_txt}")

        text = input_txt.read_text(encoding="utf-8")
        predictions = self.extract_text(
            text,
            document_id=document_id,
            output_dir=output_dir,
        )

        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with output_jsonl.open("w", encoding="utf-8") as handle:
            for prediction in predictions:
                handle.write(json.dumps(prediction.to_dict(), ensure_ascii=False) + "\n")

        return output_jsonl

    @staticmethod
    def read_predictions_from_bioc(xml_path: PathLike) -> list[ExtractionResult]:
        """
        Read CellExLink predictions from a normalized BioC XML file.

        This parser is intentionally permissive because model names can appear
        as prefixes in BioC infon keys, for example:

            CellExLink-Sapbert_id_0
            CellExLink-Sapbert_identifier_name_0
            CellExLink-Sapbert_identifier_score_0
            CellExLink-Sapbert_match_source
        """
        xml_path = Path(xml_path)

        if not xml_path.is_file():
            raise FileNotFoundError(f"BioC XML file does not exist: {xml_path}")

        tree = ET.parse(xml_path)
        root = tree.getroot()

        results: list[ExtractionResult] = []

        for document in root.findall(".//document"):
            doc_id_node = document.find("id")
            document_id = doc_id_node.text if doc_id_node is not None else ""

            for passage_index, passage in enumerate(document.findall("passage")):
                passage_offset = _safe_int(_find_child_text(passage, "offset"), default=0)

                for annotation in passage.findall("annotation"):
                    mention = _find_child_text(annotation, "text") or ""

                    start: Optional[int] = None
                    end: Optional[int] = None

                    location = annotation.find("location")
                    if location is not None:
                        start = _safe_int(location.attrib.get("offset"), default=None)
                        length = _safe_int(location.attrib.get("length"), default=None)
                        if start is not None and length is not None:
                            end = start + length

                    infons = _read_infons(annotation)

                    entity_type = (
                        infons.get("type")
                        or infons.get("entity_type")
                        or infons.get("label")
                    )

                    cl_id = _find_first_infon_value_by_suffix(infons, "_id_0")
                    cl_label = _find_first_infon_value_by_suffix(
                        infons,
                        "_identifier_name_0",
                    )
                    score = _safe_float(
                        _find_first_infon_value_by_suffix(
                            infons,
                            "_identifier_score_0",
                        )
                    )
                    source = _find_first_infon_value_by_suffix(
                        infons,
                        "_match_source",
                    )

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
                        )
                    )

        return results

    @staticmethod
    def _write_text_as_bioc_xml(
        text: str,
        output_xml: PathLike,
        *,
        document_id: str = "doc0",
    ) -> Path:
        """
        Write plain text as a minimal BioC XML file.

        This lets the plain-text API reuse the same BioC-based recognition and
        normalization pipeline.
        """
        output_xml = Path(output_xml)
        output_xml.parent.mkdir(parents=True, exist_ok=True)

        escaped_text = escape(text)
        escaped_doc_id = escape(document_id)

        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<collection>
  <source>CellExLink</source>
  <date></date>
  <key>cell_type_extraction</key>
  <document>
    <id>{escaped_doc_id}</id>
    <passage>
      <infon key="type">text</infon>
      <offset>0</offset>
      <text>{escaped_text}</text>
    </passage>
  </document>
</collection>
"""
        output_xml.write_text(xml, encoding="utf-8")
        return output_xml


def _read_infons(element: ET.Element) -> dict[str, str]:
    infons: dict[str, str] = {}

    for infon in element.findall("infon"):
        key = infon.attrib.get("key")
        if key:
            infons[key] = infon.text or ""

    return infons


def _find_child_text(element: ET.Element, child_name: str) -> Optional[str]:
    child = element.find(child_name)
    if child is None:
        return None
    return child.text


def _find_first_infon_value_by_suffix(
    infons: dict[str, str],
    suffix: str,
) -> Optional[str]:
    for key, value in infons.items():
        if key.endswith(suffix):
            return value
    return None


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def write_predictions_jsonl(
    predictions: Iterable[ExtractionResult],
    output_path: PathLike,
) -> Path:
    """
    Write extraction results to JSONL.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction.to_dict(), ensure_ascii=False) + "\n")

    return output_path