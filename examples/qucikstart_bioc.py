"""Run CellExLink on a BioC XML file.

Usage from the repository root:

    python examples/quickstart_bioc.py

Optional local-model usage after running `cellexlink download-models`:

    python examples/quickstart_bioc.py \
        --ner-model models/CellExLink-bioformer16L \
        --nen-model models/CellExLink-Sapbert
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from cellexlink import CellExLinkPipeline
from cellexlink.pipeline import DEFAULT_NEN_MODEL, DEFAULT_NER_MODEL, write_predictions_jsonl


EXAMPLES_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_DIR.parent


DEFAULT_INPUT = EXAMPLES_DIR / "sample_input.xml"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "quickstart_bioc_normalized.xml"
DEFAULT_NER_OUTPUT = REPO_ROOT / "outputs" / "quickstart_bioc_ner_predictions.xml"
DEFAULT_JSONL_OUTPUT = REPO_ROOT / "outputs" / "quickstart_bioc_predictions.jsonl"
DEFAULT_WORK_DIR = REPO_ROOT / "outputs" / "quickstart_bioc_work"


ENV_NER_MODEL = os.environ.get("CELLEXLINK_NER_MODEL", DEFAULT_NER_MODEL)
ENV_NEN_MODEL = os.environ.get("CELLEXLINK_NEN_MODEL", DEFAULT_NEN_MODEL)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CellExLink on examples/sample_input.xml and write normalized BioC XML.",
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help=f"BioC XML input file. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Normalized BioC XML output file. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--ner-output",
        default=str(DEFAULT_NER_OUTPUT),
        help=f"Intermediate NER-only BioC XML output file. Default: {DEFAULT_NER_OUTPUT}",
    )
    parser.add_argument(
        "--jsonl-output",
        default=str(DEFAULT_JSONL_OUTPUT),
        help=f"Optional JSONL summary output. Default: {DEFAULT_JSONL_OUTPUT}",
    )
    parser.add_argument(
        "--work-dir",
        default=str(DEFAULT_WORK_DIR),
        help=f"Intermediate output directory. Default: {DEFAULT_WORK_DIR}",
    )
    parser.add_argument(
        "--ner-model",
        default=ENV_NER_MODEL,
        help=(
            "NER model name or local path. "
            "Can also be set with CELLEXLINK_NER_MODEL. "
            f"Default: {ENV_NER_MODEL}"
        ),
    )
    parser.add_argument(
        "--nen-model",
        default=ENV_NEN_MODEL,
        help=(
            "Normalization model name or local path. "
            "Can also be set with CELLEXLINK_NEN_MODEL. "
            f"Default: {ENV_NEN_MODEL}"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Prediction batch size. Default: 16",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use fp16 prediction when supported.",
    )
    parser.add_argument(
        "--print-results",
        action="store_true",
        help="Print a compact prediction summary after the run finishes.",
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    input_xml = Path(args.input)
    output_xml = Path(args.output)
    ner_output_xml = Path(args.ner_output)
    jsonl_output = Path(args.jsonl_output)
    work_dir = Path(args.work_dir)

    if not input_xml.is_file():
        raise FileNotFoundError(f"Input BioC XML file does not exist: {input_xml}")

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    ner_output_xml.parent.mkdir(parents=True, exist_ok=True)
    jsonl_output.parent.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    pipeline = CellExLinkPipeline.from_pretrained(
        ner_model=args.ner_model,
        nen_model=args.nen_model,
        output_dir=work_dir,
        batch_size=args.batch_size,
        fp16=args.fp16,
    )

    pipeline.extract_bioc(
        input_xml=input_xml,
        output_xml=output_xml,
        ner_output_xml=ner_output_xml,
        output_dir=work_dir,
    )

    predictions = pipeline.read_predictions_from_bioc(output_xml)
    write_predictions_jsonl(predictions, jsonl_output)

    print(f"Input BioC XML: {input_xml}")
    print(f"Intermediate NER BioC XML: {ner_output_xml}")
    print(f"Normalized BioC XML: {output_xml}")
    print(f"JSONL summary: {jsonl_output}")
    print(f"Intermediate files: {work_dir}")

    if args.print_results:
        for prediction in predictions:
            print(json.dumps(prediction.to_dict(), indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
