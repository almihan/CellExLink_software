"""
Lightweight Docker smoke test for CellExLink.

This script is designed to run inside the Docker image without downloading large
model checkpoints. It checks imports, the CLI, shared BioC I/O utilities, and the
top-level pipeline wiring by injecting fake NER and normalization modules.

Run from the repository root image with:

    docker build -t cellexlink:latest .
    docker run --rm cellexlink:latest python /app/docker/docker_test.py

Optional real-model test, only when checkpoints are available or internet access
is intentionally allowed:

    docker run --rm \
      -v "$PWD/models:/models" \
      cellexlink:latest \
      python /app/docker/docker_test.py \
        --run-real-models \
        --ner-model /models/CellExLink-bioformer16L \
        --nen-model /models/CellExLink-Sapbert
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


DEFAULT_TEXT = "The mesothelial cell was detected near SMC clusters."


def run_command(command: list[str], *, cwd: str | Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run a command and raise a detailed error if it fails."""
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"  command: {' '.join(command)}\n"
            f"  exit code: {result.returncode}\n"
            f"  stdout:\n{result.stdout}\n"
            f"  stderr:\n{result.stderr}\n"
        )
    return result


def print_check(name: str, details: str | None = None) -> None:
    """Print a compact success line."""
    if details:
        print(f"✓ {name}: {details}")
    else:
        print(f"✓ {name}")


def verify_imports() -> dict[str, Any]:
    """Verify that public package imports are available."""
    import cellexlink
    from cellexlink import CellExLinkPipeline, ExtractionResult
    from cellexlink.io import EntitySpan, PassageRecord

    assert hasattr(cellexlink, "__version__")
    assert CellExLinkPipeline is not None
    assert ExtractionResult is not None
    assert EntitySpan is not None
    assert PassageRecord is not None

    print_check("package imports", f"cellexlink {cellexlink.__version__}")
    return {"cellexlink_version": cellexlink.__version__}


def verify_cli() -> dict[str, Any]:
    """Verify that the installed CLI and module CLI help commands work."""
    installed = run_command(["cellexlink", "--help"])
    assert "predict-bioc" in installed.stdout
    assert "predict-text" in installed.stdout
    assert "normalize-bioc" in installed.stdout

    module = run_command([sys.executable, "-m", "cellexlink.cli", "--help"])
    assert "download-models" in module.stdout

    print_check("CLI help", "cellexlink --help and python -m cellexlink.cli --help")
    return {
        "installed_cli_contains_predict_bioc": "predict-bioc" in installed.stdout,
        "module_cli_contains_download_models": "download-models" in module.stdout,
    }


def verify_bioc_io(tmp_dir: Path) -> dict[str, Any]:
    """Verify minimal BioC creation, parsing, and JSONL conversion."""
    from cellexlink.io import convert_bioc_to_jsonl, iter_bioc_passages, write_text_as_bioc

    xml_path = tmp_dir / "sample_input.xml"
    jsonl_path = tmp_dir / "sample_passages.jsonl"

    write_text_as_bioc(
        text=DEFAULT_TEXT,
        output_xml=xml_path,
        document_id="docker-smoke-doc",
    )

    passages = list(iter_bioc_passages(xml_path))
    assert len(passages) == 1
    assert passages[0].document_id == "docker-smoke-doc"
    assert "mesothelial cell" in passages[0].text

    count = convert_bioc_to_jsonl(
        srcs=xml_path,
        output_jsonl=jsonl_path,
        include_entities=False,
    )
    assert count == 1
    assert jsonl_path.is_file()

    row = json.loads(jsonl_path.read_text(encoding="utf-8").strip())
    assert row["document_id"] == "docker-smoke-doc"
    assert "text" in row

    print_check("BioC I/O", f"created {xml_path.name} and {jsonl_path.name}")
    return {"passage_count": count, "jsonl_path": str(jsonl_path)}


def install_fake_pipeline_modules() -> None:
    """
    Inject fake recognition and normalization modules.

    The real pipeline imports these functions lazily:

        from cellexlink.recognition.predict import predict_ner
        from cellexlink.normalization.linker import normalize_bioc

    By placing fake modules into sys.modules before calling the pipeline, this
    test validates pipeline wiring without importing transformers or loading
    model checkpoints.
    """

    fake_predict_module = types.ModuleType("cellexlink.recognition.predict")
    fake_linker_module = types.ModuleType("cellexlink.normalization.linker")

    def fake_predict_ner(
        model_path: str | Path,
        input_xml: str | Path,
        output_dir: str | Path,
        output_xml: str | Path,
        **_: Any,
    ) -> int:
        del model_path, output_dir

        input_xml = Path(input_xml)
        output_xml = Path(output_xml)
        output_xml.parent.mkdir(parents=True, exist_ok=True)

        tree = ET.parse(input_xml)
        root = tree.getroot()

        for passage in root.findall(".//passage"):
            text_node = passage.find("text")
            if text_node is None or text_node.text is None:
                continue

            text = text_node.text
            mention = "mesothelial cell"
            relative_start = text.find(mention)
            if relative_start < 0:
                continue

            passage_offset = 0
            offset_node = passage.find("offset")
            if offset_node is not None and offset_node.text:
                try:
                    passage_offset = int(offset_node.text)
                except ValueError:
                    passage_offset = 0

            annotation_id = f"T{len(passage.findall('annotation')) + 1}"
            annotation = ET.SubElement(passage, "annotation", {"id": annotation_id})
            ET.SubElement(annotation, "infon", {"key": "type"}).text = "cell_type"
            ET.SubElement(
                annotation,
                "location",
                {
                    "offset": str(passage_offset + relative_start),
                    "length": str(len(mention)),
                },
            )
            ET.SubElement(annotation, "text").text = mention

        tree.write(output_xml, encoding="utf-8", xml_declaration=True)
        return 0

    def fake_normalize_bioc(
        input_xml: str | Path,
        output_xml: str | Path,
        **_: Any,
    ) -> Path:
        input_xml = Path(input_xml)
        output_xml = Path(output_xml)
        output_xml.parent.mkdir(parents=True, exist_ok=True)

        tree = ET.parse(input_xml)
        root = tree.getroot()

        for annotation in root.findall(".//annotation"):
            ET.SubElement(annotation, "infon", {"key": "CellExLink-Sapbert_id_0"}).text = "CL:0000077"
            ET.SubElement(
                annotation,
                "infon",
                {"key": "CellExLink-Sapbert_identifier_name_0"},
            ).text = "mesothelial cell"
            ET.SubElement(
                annotation,
                "infon",
                {"key": "CellExLink-Sapbert_identifier_score_0"},
            ).text = "0.99"
            ET.SubElement(
                annotation,
                "infon",
                {"key": "CellExLink-Sapbert_match_source"},
            ).text = "docker-smoke-test"

        tree.write(output_xml, encoding="utf-8", xml_declaration=True)
        return output_xml

    fake_predict_module.predict_ner = fake_predict_ner  # type: ignore[attr-defined]
    fake_linker_module.normalize_bioc = fake_normalize_bioc  # type: ignore[attr-defined]

    sys.modules["cellexlink.recognition.predict"] = fake_predict_module
    sys.modules["cellexlink.normalization.linker"] = fake_linker_module


def verify_pipeline_smoke(tmp_dir: Path) -> dict[str, Any]:
    """Verify the top-level pipeline wiring without loading real models."""
    install_fake_pipeline_modules()

    from cellexlink import CellExLinkPipeline

    pipeline = CellExLinkPipeline.from_pretrained(
        ner_model="docker-fake-ner-model",
        nen_model="docker-fake-nen-model",
        output_dir=tmp_dir / "pipeline_work",
    )

    results = pipeline.extract_text(
        DEFAULT_TEXT,
        document_id="docker-smoke-doc",
        output_dir=tmp_dir / "pipeline_work",
    )

    assert len(results) == 1
    assert results[0].mention == "mesothelial cell"
    assert results[0].cl_id == "CL:0000077"
    assert results[0].cl_label == "mesothelial cell"
    assert results[0].source == "docker-smoke-test"

    print_check("pipeline smoke test", f"{results[0].mention} -> {results[0].cl_id}")
    return {"prediction_count": len(results), "first_prediction": results[0].to_dict()}


def verify_real_models(args: argparse.Namespace, tmp_dir: Path) -> dict[str, Any]:
    """
    Optionally run a real end-to-end model test.

    This is disabled by default because it may download checkpoints, require more
    memory, or take several minutes on CPU.
    """
    if not args.run_real_models:
        print_check("real model test skipped", "use --run-real-models to enable")
        return {"skipped": True}

    from cellexlink import CellExLinkPipeline

    pipeline = CellExLinkPipeline.from_pretrained(
        ner_model=args.ner_model,
        nen_model=args.nen_model,
        output_dir=tmp_dir / "real_model_work",
        batch_size=args.batch_size,
        fp16=args.fp16,
    )

    results = pipeline.extract_text(
        DEFAULT_TEXT,
        document_id="docker-real-model-doc",
        output_dir=tmp_dir / "real_model_work",
    )

    print_check("real model test", f"received {len(results)} predictions")
    return {"skipped": False, "prediction_count": len(results)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight CellExLink Docker smoke tests.")
    parser.add_argument(
        "--run-real-models",
        action="store_true",
        help="Run actual NER and NEN models. Disabled by default.",
    )
    parser.add_argument(
        "--ner-model",
        default="almire/CellExLink-bioformer16L",
        help="NER model name or local path for --run-real-models.",
    )
    parser.add_argument(
        "--nen-model",
        default="almire/CellExLink-Sapbert",
        help="NEN model name or local path for --run-real-models.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for optional real-model test.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use fp16 for optional real-model test when supported.",
    )
    parser.add_argument(
        "--json-summary",
        default=None,
        help="Optional path for a JSON summary of test results.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    summary: dict[str, Any] = {}

    try:
        with tempfile.TemporaryDirectory(prefix="cellexlink_docker_test_") as tmp:
            tmp_dir = Path(tmp)
            summary["imports"] = verify_imports()
            summary["cli"] = verify_cli()
            summary["bioc_io"] = verify_bioc_io(tmp_dir)
            summary["pipeline_smoke"] = verify_pipeline_smoke(tmp_dir)
            summary["real_models"] = verify_real_models(args, tmp_dir)

        if args.json_summary:
            summary_path = Path(args.json_summary)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print_check("JSON summary", str(summary_path))

        print("\nAll requested Docker smoke tests passed.")
        return 0
    except Exception as exc:
        print(f"\nDocker smoke test failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
