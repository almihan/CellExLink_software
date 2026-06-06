"""Tests for the top-level CellExLink command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cellexlink.pipeline import ExtractionResult


def test_cli_help_lists_main_commands(capsys: pytest.CaptureFixture[str]) -> None:
    from cellexlink.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "predict-text" in captured.out
    assert "predict-bioc" in captured.out
    assert "normalize-bioc" in captured.out
    assert "download-models" in captured.out


def test_predict_text_parser_accepts_text_input(tmp_path: Path) -> None:
    from cellexlink.cli import build_parser

    output = tmp_path / "predictions.jsonl"
    parser = build_parser()
    args = parser.parse_args(
        [
            "predict-text",
            "--text",
            "The mesothelial cell was detected.",
            "--output",
            str(output),
        ]
    )

    assert args.command == "predict-text"
    assert args.text == "The mesothelial cell was detected."
    assert args.output == str(output)


def test_predict_bioc_parser_accepts_paths(tmp_path: Path) -> None:
    from cellexlink.cli import build_parser

    input_xml = tmp_path / "input.xml"
    output_xml = tmp_path / "normalized.xml"
    parser = build_parser()
    args = parser.parse_args(
        [
            "predict-bioc",
            "--input",
            str(input_xml),
            "--output",
            str(output_xml),
            "--ner-model",
            "local-ner",
            "--nen-model",
            "local-nen",
        ]
    )

    assert args.command == "predict-bioc"
    assert args.input == str(input_xml)
    assert args.output == str(output_xml)
    assert args.ner_model == "local-ner"
    assert args.nen_model == "local-nen"


def test_predict_text_cli_dispatch_can_be_tested_without_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import cellexlink.cli as cli

    output = tmp_path / "predictions.jsonl"

    class FakePipeline:
        def extract_text(self, text: str, *, document_id: str, output_dir: str):
            assert "mesothelial cell" in text
            assert document_id == "doc0"
            return [
                ExtractionResult(
                    document_id=document_id,
                    passage_index=0,
                    mention="mesothelial cell",
                    start=4,
                    end=20,
                    entity_type="cell_type",
                    cl_id="CL:0000077",
                    cl_label="mesothelial cell",
                    score=0.99,
                    source="unit-test",
                )
            ]

    monkeypatch.setattr(cli, "make_pipeline_from_args", lambda args: FakePipeline())

    exit_code = cli.main(
        [
            "predict-text",
            "--text",
            "The mesothelial cell was detected.",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert "CellExLink text predictions written to" in capsys.readouterr().out
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "document_id": "doc0",
            "passage_index": 0,
            "mention": "mesothelial cell",
            "start": 4,
            "end": 20,
            "entity_type": "cell_type",
            "cl_id": "CL:0000077",
            "cl_label": "mesothelial cell",
            "score": 0.99,
            "source": "unit-test",
        }
    ]
