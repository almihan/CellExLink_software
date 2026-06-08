"""Model-free smoke tests for the CellExLink end-to-end pipeline."""

from __future__ import annotations

import shutil
import sys
import types
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest


def _add_infon(element: ET.Element, key: str, value: str) -> None:
    infon = ET.SubElement(element, "infon", {"key": key})
    infon.text = value


def _install_fake_model_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install fake recognition/normalization modules so tests need no models."""
    fake_recognition_predict = types.ModuleType("cellexlink.recognition.predict")

    def fake_predict_ner(
        *,
        model_path,
        input_xml,
        output_dir,
        output_xml,
        warmup_runs=1,
        per_device_predict_batch_size=16,
        fp16=False,
        trust_remote_code=False,
        **kwargs,
    ) -> int:
        del model_path, output_dir, warmup_runs, per_device_predict_batch_size, fp16, trust_remote_code, kwargs

        tree = ET.parse(input_xml)
        root = tree.getroot()
        annotation_counter = 1
        for passage in root.findall(".//passage"):
            text_node = passage.find("text")
            offset_node = passage.find("offset")
            if text_node is None or not text_node.text:
                continue
            passage_text = text_node.text
            passage_offset = int(offset_node.text) if offset_node is not None and offset_node.text else 0
            mention = "mesothelial cell"
            local_start = passage_text.find(mention)
            if local_start < 0:
                continue

            annotation = ET.SubElement(passage, "annotation", {"id": f"T{annotation_counter}"})
            _add_infon(annotation, "type", "cell_type")
            ET.SubElement(
                annotation,
                "location",
                {
                    "offset": str(passage_offset + local_start),
                    "length": str(len(mention)),
                },
            )
            mention_node = ET.SubElement(annotation, "text")
            mention_node.text = mention
            annotation_counter += 1

        Path(output_xml).parent.mkdir(parents=True, exist_ok=True)
        ET.indent(tree, space="  ")
        tree.write(output_xml, encoding="utf-8", xml_declaration=True)
        return 0

    fake_recognition_predict.predict_ner = fake_predict_ner
    monkeypatch.setitem(sys.modules, "cellexlink.recognition.predict", fake_recognition_predict)

    fake_normalization_linker = types.ModuleType("cellexlink.normalization.linker")

    def fake_normalize_bioc(input_xml, output_xml, **kwargs) -> None:
        del kwargs
        tree = ET.parse(input_xml)
        for annotation in tree.getroot().findall(".//annotation"):
            _add_infon(annotation, "CellExLink-Sapbert_id_0", "CL:0000077")
            _add_infon(annotation, "CellExLink-Sapbert_identifier_name_0", "mesothelial cell")
            _add_infon(annotation, "CellExLink-Sapbert_identifier_score_0", "0.99")
            _add_infon(annotation, "CellExLink-Sapbert_match_source", "unit-test")
        Path(output_xml).parent.mkdir(parents=True, exist_ok=True)
        ET.indent(tree, space="  ")
        tree.write(output_xml, encoding="utf-8", xml_declaration=True)

    fake_normalization_linker.normalize_bioc = fake_normalize_bioc
    monkeypatch.setitem(sys.modules, "cellexlink.normalization.linker", fake_normalization_linker)


def test_extract_text_smoke_without_loading_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cellexlink import CellExLinkPipeline

    _install_fake_model_modules(monkeypatch)

    pipeline = CellExLinkPipeline.from_pretrained(
        ner_model="dummy-ner-model",
        nen_model="dummy-nen-model",
        output_dir=tmp_path / "work",
    )

    results = pipeline.extract_text(
        "The mesothelial cell was detected in the sample.",
        document_id="doc-smoke",
        output_dir=tmp_path / "text-work",
    )

    assert len(results) == 1
    assert results[0].document_id == "doc-smoke"
    assert results[0].mention == "mesothelial cell"
    assert results[0].start == 4
    assert results[0].end == 20
    assert results[0].entity_type == "cell_type"
    assert results[0].cl_id == "CL:0000077"
    assert results[0].cl_label == "mesothelial cell"
    assert results[0].score == pytest.approx(0.99)
    assert results[0].source == "unit-test"


def test_extract_bioc_smoke_without_loading_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cellexlink import CellExLinkPipeline
    from cellexlink.io import write_text_as_bioc_xml

    _install_fake_model_modules(monkeypatch)

    input_xml = tmp_path / "input.xml"
    output_xml = tmp_path / "normalized.xml"
    ner_xml = tmp_path / "ner.xml"
    write_text_as_bioc_xml(
        "The mesothelial cell was detected in the sample.",
        input_xml,
        document_id="doc-bioc-smoke",
    )

    pipeline = CellExLinkPipeline.from_pretrained(
        ner_model="dummy-ner-model",
        nen_model="dummy-nen-model",
        output_dir=tmp_path / "work",
    )

    returned_path = pipeline.extract_bioc(
        input_xml=input_xml,
        output_xml=output_xml,
        ner_output_xml=ner_xml,
        output_dir=tmp_path / "work",
    )

    assert returned_path == output_xml
    assert output_xml.is_file()
    assert ner_xml.is_file()

    results = pipeline.read_predictions_from_bioc(output_xml)
    assert len(results) == 1
    assert results[0].document_id == "doc-bioc-smoke"
    assert results[0].mention == "mesothelial cell"
    assert results[0].cl_id == "CL:0000077"


def test_extract_bioc_raises_for_missing_input(tmp_path: Path) -> None:
    from cellexlink import CellExLinkPipeline

    pipeline = CellExLinkPipeline.from_pretrained(
        ner_model="dummy-ner-model",
        nen_model="dummy-nen-model",
        output_dir=tmp_path / "work",
    )

    with pytest.raises(FileNotFoundError):
        pipeline.extract_bioc(
            input_xml=tmp_path / "missing.xml",
            output_xml=tmp_path / "out.xml",
        )

def test_public_api_surface_is_obvious_without_loading_models() -> None:
    """The public package should expose NER, NEN, and end-to-end APIs.

    This test must not load Bioformer, SapBERT, Ab3P, or ontology resources.
    It only checks that the public API shape is stable.
    """

    from cellexlink import CellExLinkPipeline, ExtractionResult

    pipe = CellExLinkPipeline.from_pretrained(
        ner_model="dummy-ner-model",
        nen_model="dummy-nen-model",
    )

    expected_methods = [
        "recognize_text",      # NER only, plain text
        "normalize_mentions",  # NEN only, detected/gold mentions
        "extract_text",        # end-to-end, plain text
        "recognize_bioc",      # NER only, BioC XML
        "normalize_bioc",      # NEN only, BioC XML
        "extract_bioc",        # end-to-end, BioC XML
    ]

    for method_name in expected_methods:
        assert hasattr(pipe, method_name), f"Missing public API method: {method_name}"
        assert callable(getattr(pipe, method_name))

    result = ExtractionResult(
        document_id="doc0",
        passage_index=0,
        mention="SMC",
        start=10,
        end=13,
        entity_type="cell_type",
        cl_id="CL:0000192",
        cl_label="smooth muscle cell",
        score=0.95,
        source="test",
    )

    assert result.to_dict()["mention"] == "SMC"
    assert result.to_dict()["cl_id"] == "CL:0000192"