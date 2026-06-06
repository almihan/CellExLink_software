from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from . import __version__
from .pipeline import (
    DEFAULT_NEN_MODEL,
    DEFAULT_NER_MODEL,
    CellExLinkPipeline,
    write_predictions_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cellexlink",
        description=(
            "CellExLink: end-to-end cell-type recognition and "
            "Cell Ontology normalization from biomedical text."
        ),
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"cellexlink {__version__}",
    )

    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        required=True,
    )

    add_predict_bioc_parser(subparsers)
    add_normalize_bioc_parser(subparsers)
    add_predict_text_parser(subparsers)
    add_download_models_parser(subparsers)

    return parser


def add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ner-model",
        default=DEFAULT_NER_MODEL,
        help=(
            "NER model name or local path. "
            f"Default: {DEFAULT_NER_MODEL}"
        ),
    )

    parser.add_argument(
        "--nen-model",
        default=DEFAULT_NEN_MODEL,
        help=(
            "NEN/linking model name or local path. "
            f"Default: {DEFAULT_NEN_MODEL}"
        ),
    )

    parser.add_argument(
        "--ontology-path",
        default=None,
        help=(
            "Path to Cell Ontology JSONL resource. "
            "If omitted, the package/default normalization setting is used."
        ),
    )

    parser.add_argument(
        "--abbreviations-path",
        default=None,
        help=(
            "Path to abbreviation TSV file. "
            "If omitted, the package/default normalization setting is used."
        ),
    )

    parser.add_argument(
        "--disable-abbreviations",
        action="store_true",
        help="Disable abbreviation dictionary and document-level abbreviation handling.",
    )

    parser.add_argument(
        "--output-dir",
        default="cellexlink_outputs",
        help="Directory for intermediate files and model outputs.",
    )

    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Number of warmup runs used by model prediction code.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Prediction batch size.",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use fp16 prediction when supported.",
    )

    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to Hugging Face model loading when needed.",
    )


def add_predict_bioc_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "predict-bioc",
        help="Run NER + Cell Ontology normalization on a BioC XML file.",
    )

    add_common_model_args(parser)

    parser.add_argument(
        "--input",
        required=True,
        help="Input BioC XML file.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Final normalized BioC XML output file.",
    )

    parser.add_argument(
        "--ner-output",
        default=None,
        help="Optional intermediate NER-only BioC XML output file.",
    )

    parser.set_defaults(func=cmd_predict_bioc)


def add_normalize_bioc_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "normalize-bioc",
        help=(
            "Run only Cell Ontology normalization on a BioC XML file that "
            "already contains cell-type annotations."
        ),
    )

    add_common_model_args(parser)

    parser.add_argument(
        "--input",
        required=True,
        help="Input BioC XML file containing cell-type annotations.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Normalized BioC XML output file.",
    )

    parser.set_defaults(func=cmd_normalize_bioc)


def add_predict_text_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "predict-text",
        help="Run end-to-end CellExLink on plain text.",
    )

    add_common_model_args(parser)

    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument(
        "--text",
        default=None,
        help="Plain text string to process.",
    )

    input_group.add_argument(
        "--input",
        default=None,
        help="Plain text file to process.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file.",
    )

    parser.add_argument(
        "--document-id",
        default="doc0",
        help="Document identifier to use for plain-text input.",
    )

    parser.set_defaults(func=cmd_predict_text)


def add_download_models_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "download-models",
        help="Download the default CellExLink Hugging Face checkpoints.",
    )

    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory where models should be downloaded.",
    )

    parser.add_argument(
        "--ner-model",
        default=DEFAULT_NER_MODEL,
        help=f"NER model repo ID. Default: {DEFAULT_NER_MODEL}",
    )

    parser.add_argument(
        "--nen-model",
        default=DEFAULT_NEN_MODEL,
        help=f"NEN model repo ID. Default: {DEFAULT_NEN_MODEL}",
    )

    parser.set_defaults(func=cmd_download_models)


def make_pipeline_from_args(args: argparse.Namespace) -> CellExLinkPipeline:
    return CellExLinkPipeline.from_pretrained(
        ner_model=args.ner_model,
        nen_model=args.nen_model,
        ontology_path=args.ontology_path,
        abbreviations_path=args.abbreviations_path,
        disable_abbreviations=args.disable_abbreviations,
        output_dir=args.output_dir,
        warmup_runs=args.warmup_runs,
        batch_size=args.batch_size,
        fp16=args.fp16,
        trust_remote_code=args.trust_remote_code,
    )


def cmd_predict_bioc(args: argparse.Namespace) -> int:
    pipeline = make_pipeline_from_args(args)

    output_path = pipeline.extract_bioc(
        input_xml=args.input,
        output_xml=args.output,
        ner_output_xml=args.ner_output,
        output_dir=args.output_dir,
    )

    print(f"CellExLink BioC prediction written to: {output_path}")
    return 0


def cmd_normalize_bioc(args: argparse.Namespace) -> int:
    pipeline = make_pipeline_from_args(args)

    output_path = pipeline.normalize_bioc(
        input_xml=args.input,
        output_xml=args.output,
    )

    print(f"CellExLink normalized BioC output written to: {output_path}")
    return 0


def cmd_predict_text(args: argparse.Namespace) -> int:
    pipeline = make_pipeline_from_args(args)

    if args.text is not None:
        predictions = pipeline.extract_text(
            args.text,
            document_id=args.document_id,
            output_dir=args.output_dir,
        )
        output_path = write_predictions_jsonl(predictions, args.output)
    else:
        output_path = pipeline.extract_text_file(
            input_txt=args.input,
            output_jsonl=args.output,
            document_id=args.document_id,
            output_dir=args.output_dir,
        )

    print(f"CellExLink text predictions written to: {output_path}")
    return 0


def cmd_download_models(args: argparse.Namespace) -> int:
    """
    Download model checkpoints from Hugging Face.

    This command keeps large model files outside the GitHub repository while
    still making setup easy for reviewers and users.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for model download. "
            "Install it with: pip install huggingface-hub"
        ) from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ner_target = output_dir / Path(args.ner_model).name
    nen_target = output_dir / Path(args.nen_model).name

    print(f"Downloading NER model {args.ner_model} to {ner_target}")
    snapshot_download(
        repo_id=args.ner_model,
        local_dir=str(ner_target),
        local_dir_use_symlinks=False,
    )

    print(f"Downloading NEN model {args.nen_model} to {nen_target}")
    snapshot_download(
        repo_id=args.nen_model,
        local_dir=str(nen_target),
        local_dir_use_symlinks=False,
    )

    manifest = {
        "ner_model": {
            "repo_id": args.ner_model,
            "local_dir": str(ner_target),
        },
        "nen_model": {
            "repo_id": args.nen_model,
            "local_dir": str(nen_target),
        },
    }

    manifest_path = output_dir / "models.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"Model manifest written to: {manifest_path}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())