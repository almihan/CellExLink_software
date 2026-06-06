"""Tests for shared BioC and JSONL I/O utilities."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET


def test_write_text_as_bioc_and_convert_to_jsonl(tmp_path: Path) -> None:
    from cellexlink.io import (
        collection_summary,
        convert_bioc_to_jsonl,
        iter_bioc_passage_records,
        read_jsonl,
        write_text_as_bioc_xml,
    )

    text = "The mesothelial cell and SMC clusters were observed."
    xml_path = tmp_path / "sample.xml"
    jsonl_path = tmp_path / "passages.jsonl"

    written = write_text_as_bioc_xml(text, xml_path, document_id="doc0")

    assert written == xml_path
    assert xml_path.is_file()
    assert collection_summary(xml_path) == {
        "documents": 1,
        "passages": 1,
        "annotations": 0,
    }

    records = list(iter_bioc_passage_records([xml_path], include_entities=False))
    assert len(records) == 1
    assert records[0].document_id == "doc0"
    assert records[0].passage_id == 0
    assert records[0].passage_offset == 0
    assert records[0].text == text
    assert records[0].entities == []

    convert_bioc_to_jsonl([xml_path], jsonl_path, include_entities=False)
    rows = read_jsonl(jsonl_path)
    assert rows[0]["document_id"] == "doc0"
    assert rows[0]["text"] == text
    assert "entities" not in rows[0]


def test_write_predictions_to_bioc_and_read_annotations(tmp_path: Path) -> None:
    from cellexlink.io import (
        PredictedEntity,
        collection_summary,
        read_bioc_annotations,
        read_infons,
        write_predictions_to_bioc_xml,
        write_text_as_bioc_xml,
    )

    text = "The mesothelial cell and SMC clusters were observed."
    xml_path = tmp_path / "sample.xml"
    output_xml = tmp_path / "ner_predictions.xml"
    write_text_as_bioc_xml(text, xml_path, document_id="doc0")

    start = text.index("mesothelial cell")
    entity = PredictedEntity(
        document_id="doc0",
        passage_id=0,
        start=start,
        end=start + len("mesothelial cell"),
        label="cell_type",
        text="mesothelial cell",
        score=0.98765,
        infons={"model": "unit-test"},
    )

    written = write_predictions_to_bioc_xml(
        input_xml=xml_path,
        output_xml=output_xml,
        predicted_entities=[entity],
    )

    assert written == output_xml
    assert collection_summary(output_xml)["annotations"] == 1

    annotations = read_bioc_annotations(output_xml)
    assert len(annotations) == 1
    assert annotations[0].document_id == "doc0"
    assert annotations[0].start == start
    assert annotations[0].end == start + len("mesothelial cell")
    assert annotations[0].text == "mesothelial cell"
    assert annotations[0].label == "cell_type"

    tree = ET.parse(output_xml)
    annotation = tree.getroot().find(".//annotation")
    assert annotation is not None
    infons = read_infons(annotation)
    assert infons["type"] == "cell_type"
    assert infons["model"] == "unit-test"
    assert "score" in infons


def test_bioc_annotation_offsets_are_absolute(tmp_path: Path) -> None:
    from cellexlink.io import PredictedEntity, read_bioc_annotations, write_predictions_to_bioc_xml

    xml_path = tmp_path / "offset_sample.xml"
    xml_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<collection>
  <source>unit-test</source>
  <date></date>
  <key>offset-test</key>
  <document>
    <id>doc-with-offset</id>
    <passage>
      <infon key="type">abstract</infon>
      <offset>100</offset>
      <text>SMC clusters were observed.</text>
    </passage>
  </document>
</collection>
""",
        encoding="utf-8",
    )

    output_xml = tmp_path / "offset_predictions.xml"
    write_predictions_to_bioc_xml(
        xml_path,
        output_xml,
        predicted_entities=[
            PredictedEntity(
                document_id="doc-with-offset",
                passage_id=0,
                start=100,
                end=103,
                label="cell_type",
                text="SMC",
            )
        ],
    )

    annotations = read_bioc_annotations(output_xml)
    assert len(annotations) == 1
    assert annotations[0].start == 100
    assert annotations[0].end == 103
    assert annotations[0].text == "SMC"
