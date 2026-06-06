"""Run CellExLink on a plain-text file.

Usage from the repository root:

    python examples/quickstart_text.py

Optional local-model usage after running `cellexlink download-models`:

    python examples/quickstart_text.py \
        --ner-model models/CellExLink-bioformer16L \
        --nen-model models/CellExLink-Sapbert
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from cellexlink import CellExLinkPipeline
from cellexlink.pipeline import DEFAULT_NEN_MODEL, DEFAULT_NER_MODEL


EXAMPLES_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_DIR.parent


DEFAULT_INPUT = EXAMPLES_DIR / "sample_input.txt"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "quickstart_text_predictions.jsonl"
DEFAULT_WORK_DIR = REPO_ROOT / "outputs" / "quickstart_text_work"


# Environment variables make the example convenient on clusters or containers.
ENV_NER_MODEL = os.environ.get("CELLEXLINK_NER_MODEL", DEFAULT_NER_MODEL)
ENV_NEN_MODEL = os.environ.get("CELLEXLINK_NEN_MODEL", DEFAULT_NEN_MODEL)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CellExLink on examples/sample_input.txt and write JSONL output.",
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help=f"Plain-text input file. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"JSONL output file. Default: {DEFAULT_OUTPUT}",
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
        help="Print JSONL predictions after the run finishes.",
    )

    return parser


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> int:
    args = build_parser().parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    work_dir = Path(args.work_dir)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input text file does not exist: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    pipeline = CellExLinkPipeline.from_pretrained(
        ner_model=args.ner_model,
        nen_model=args.nen_model,
        output_dir=work_dir,
        batch_size=args.batch_size,
        fp16=args.fp16,
    )

    pipeline.extract_text_file(
        input_txt=input_path,
        output_jsonl=output_path,
        document_id="sample-text",
        output_dir=work_dir,
    )

    print(f"Input text: {input_path}")
    print(f"Prediction JSONL: {output_path}")
    print(f"Intermediate files: {work_dir}")

    if args.print_results:
        for item in iter_jsonl(output_path):
            print(json.dumps(item, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
