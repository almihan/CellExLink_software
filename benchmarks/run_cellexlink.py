#!/usr/bin/env python
"""Run CellExLink prediction files for the SoftwareX benchmark table.

This runner is intentionally small.  It only creates the prediction BioC XML
files needed before evaluation:

1. ``--mode ner``: NER-only predictions for the NER benchmark.
2. ``--mode normalize``: gold-span NEN predictions.  The input XML already
   contains gold mention spans.  Gold ``identifier`` infons are preserved in the
   output so the original-compatible evaluator can compare them with predicted
   ``CellExLink-Sapbert_id_0`` fields.
3. ``--mode full``: NER followed by NEN on predicted spans for strict
   end-to-end evaluation.

Do not strip gold ``identifier`` infons for manuscript NEN evaluation.
The normalizer must ignore those gold IDs, but the evaluator needs them.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from xml.etree import ElementTree as ET

PathLike = str | os.PathLike[str]

DEFAULT_NER_MODEL = "models/CellExLink-bioformer16L"
DEFAULT_NEN_MODEL = "models/CellExLink-Sapbert"
DEFAULT_ONTOLOGY = "src/cellexlink/resources/cell_ontology_v2025-12-17.jsonl"
DEFAULT_ABBREVIATIONS = "src/cellexlink/resources/abbreviations.tsv"

# These are predicted/system normalization fields.  Keep gold fields such as:
#   <infon key="identifier">CL:...</infon>
# because gold-span NEN evaluation needs them.
PREDICTED_NORMALIZATION_SUFFIXES = (
    "_id_0",
    "_identifier_name_0",
    "_identifier_score_0",
    "_embedding_score_0",
    "_preferred_label_0",
    "_match_source",
    "_abbreviation_method",
    "_expanded_long_form",
    "_ab3p_method",
    "_ab3p_matched_key",
    "_ab3p_match_score",
)

GENERIC_PARENT_NAMES = {
    "",
    ".",
    "data",
    "dataset",
    "datasets",
    "evaluation",
    "test",
    "train",
    "validation",
    "dev",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CellExLink before benchmark evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input BioC XML file or directory. Repeat this option for multiple datasets.",
    )
    parser.add_argument(
        "--glob",
        default="*.xml",
        help="Glob used when an --input value is a directory.",
    )
    parser.add_argument(
        "--mode",
        choices=("ner", "normalize", "full"),
        default="full",
        help=(
            "ner = recognition only; normalize = gold-span NEN only; "
            "full = recognition followed by normalization."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_outputs/cellexlink",
        help="Directory for prediction XML files and the run manifest.",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Directory for intermediate files. Defaults to OUTPUT_DIR/work.",
    )
    parser.add_argument("--manifest", default=None, help="CSV manifest path.")

    parser.add_argument("--ner-model", default=DEFAULT_NER_MODEL)
    parser.add_argument("--nen-model", default=DEFAULT_NEN_MODEL)
    parser.add_argument(
        "--ontology-path",
        default=DEFAULT_ONTOLOGY,
        help="Cell Ontology JSONL resource used by normalization.",
    )
    parser.add_argument(
        "--abbreviations-path",
        default=DEFAULT_ABBREVIATIONS,
        help="Abbreviation TSV resource used by normalization.",
    )
    parser.add_argument(
        "--disable-abbreviations",
        action="store_true",
        help="Disable abbreviation dictionary and document-level abbreviation handling.",
    )

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument(
        "--clean-normalization-input",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For --mode normalize, remove previous predicted normalization infons "
            "from a temporary copy while preserving gold identifier infons."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a dataset if the expected final output XML already exists.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later inputs after a failure and return nonzero at the end.",
    )
    return parser


def discover_xml_files(inputs: Sequence[str], pattern: str = "*.xml") -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            files.extend(sorted(path.glob(pattern)))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"Input does not exist: {path}")

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def safe_dataset_name(path: Path) -> str:
    """Create stable names such as CellLink_validation or CRAFT_test."""

    stem = _safe_name(path.stem)
    parent = _safe_name(path.parent.name)
    if parent.lower() in GENERIC_PARENT_NAMES:
        return stem
    return f"{parent}_{stem}"


def _safe_name(value: str) -> str:
    value = value.strip().replace(" ", "_")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("._-") or "dataset"


def ensure_no_duplicate_dataset_names(inputs: Sequence[Path]) -> None:
    names: dict[str, Path] = {}
    for path in inputs:
        name = safe_dataset_name(path)
        if name in names:
            raise ValueError(
                "Two input files produce the same dataset name "
                f"'{name}': {names[name]} and {path}. Rename one file or place it "
                "in a dataset-specific parent directory."
            )
        names[name] = path


def should_remove_predicted_normalization_infon(key: str) -> bool:
    # Do NOT remove gold keys such as "identifier" or "type".
    return any(key.endswith(suffix) for suffix in PREDICTED_NORMALIZATION_SUFFIXES)


def clean_normalization_input(input_xml: PathLike, output_xml: PathLike) -> Path:
    """Remove old predicted NEN fields while preserving gold IDs and spans."""

    input_path = Path(input_xml)
    output_path = Path(output_xml)
    tree = ET.parse(input_path)
    root = tree.getroot()

    for annotation in root.findall(".//annotation"):
        for infon in list(annotation.findall("infon")):
            key = infon.attrib.get("key", "")
            if should_remove_predicted_normalization_infon(key):
                annotation.remove(infon)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def count_bioc_passages(path: PathLike) -> int:
    try:
        return len(ET.parse(path).getroot().findall(".//passage"))
    except Exception:
        return 0


def count_bioc_annotations(path: PathLike) -> int:
    try:
        return len(ET.parse(path).getroot().findall(".//annotation"))
    except Exception:
        return 0


def write_csv(rows: Iterable[Mapping[str, Any]], output_csv: PathLike) -> Path:
    rows = list(rows)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def make_pipeline(args: argparse.Namespace):
    """Import lazily so --help works without ML dependencies installed."""

    # Useful when running from a fresh clone with: python benchmarks/run_cellexlink.py
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if src_dir.is_dir() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from cellexlink import CellExLinkPipeline

    return CellExLinkPipeline.from_pretrained(
        ner_model=args.ner_model,
        nen_model=args.nen_model,
        ontology_path=args.ontology_path,
        abbreviations_path=args.abbreviations_path,
        disable_abbreviations=args.disable_abbreviations,
        output_dir=args.work_dir,
        warmup_runs=args.warmup_runs,
        batch_size=args.batch_size,
        fp16=args.fp16,
        trust_remote_code=args.trust_remote_code,
    )


def expected_outputs(output_dir: Path, dataset_name: str) -> tuple[Path, Path, Path]:
    ner_xml = output_dir / f"{dataset_name}.ner.xml"
    normalized_xml = output_dir / f"{dataset_name}.normalized.xml"
    clean_xml = output_dir / "work" / dataset_name / "gold_spans.clean_for_normalization.xml"
    return ner_xml, normalized_xml, clean_xml


def run_one(
    *,
    pipeline: Any,
    args: argparse.Namespace,
    input_xml: Path,
    output_dir: Path,
    work_dir: Path,
) -> dict[str, Any]:
    dataset = safe_dataset_name(input_xml)
    ner_xml, normalized_xml, clean_xml = expected_outputs(output_dir, dataset)
    final_xml = ner_xml if args.mode == "ner" else normalized_xml

    row: dict[str, Any] = {
        "dataset": dataset,
        "mode": args.mode,
        "input_xml": str(input_xml),
        "ner_xml": str(ner_xml),
        "normalized_xml": str(normalized_xml),
        "final_output": str(final_xml),
        "ner_model": args.ner_model,
        "nen_model": args.nen_model,
        "ontology_path": args.ontology_path,
        "abbreviations_path": "" if args.disable_abbreviations else args.abbreviations_path,
        "input_passages": count_bioc_passages(input_xml),
        "input_annotations": count_bioc_annotations(input_xml),
    }

    if args.skip_existing and final_xml.exists():
        row.update(
            status="skipped_existing",
            elapsed_seconds="0.000",
            output_annotations=count_bioc_annotations(final_xml),
        )
        print(f"{dataset}: skipped existing -> {final_xml}")
        return row

    start = time.perf_counter()
    if args.mode == "ner":
        pipeline.recognize_bioc(
            input_xml=input_xml,
            output_xml=ner_xml,
            output_dir=work_dir / dataset / "ner",
        )
    elif args.mode == "normalize":
        normalization_input = input_xml
        if args.clean_normalization_input:
            normalization_input = clean_normalization_input(input_xml, clean_xml)
            row["normalization_input_xml"] = str(normalization_input)
            row["gold_identifier_infons_preserved"] = True
        pipeline.normalize_bioc(
            input_xml=normalization_input,
            output_xml=normalized_xml,
        )
    elif args.mode == "full":
        pipeline.extract_bioc(
            input_xml=input_xml,
            output_xml=normalized_xml,
            ner_output_xml=ner_xml,
            output_dir=work_dir / dataset,
        )
    else:  # pragma: no cover - argparse prevents this.
        raise ValueError(f"Unsupported mode: {args.mode}")

    elapsed = time.perf_counter() - start
    row.update(
        status="ok",
        elapsed_seconds=f"{elapsed:.3f}",
        output_annotations=count_bioc_annotations(final_xml),
    )
    print(f"{dataset}: ok in {elapsed:.2f}s -> {final_xml}")
    return row


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir = str(Path(args.work_dir) if args.work_dir else output_dir / "work")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    manifest = Path(args.manifest) if args.manifest else output_dir / "run_manifest.csv"
    inputs = discover_xml_files(args.input, args.glob)
    if not inputs:
        raise FileNotFoundError("No input BioC XML files were found.")
    ensure_no_duplicate_dataset_names(inputs)

    if args.mode == "normalize" and not args.clean_normalization_input:
        print(
            "WARNING: --no-clean-normalization-input was used. If the input XML "
            "already contains predicted normalization fields, NEN output may be stale.",
            file=sys.stderr,
        )

    pipeline = make_pipeline(args)
    rows: list[dict[str, Any]] = []
    failures = 0

    for input_xml in inputs:
        try:
            rows.append(
                run_one(
                    pipeline=pipeline,
                    args=args,
                    input_xml=input_xml,
                    output_dir=output_dir,
                    work_dir=work_dir,
                )
            )
        except Exception as exc:  # noqa: BLE001 - benchmark manifest should record failures.
            failures += 1
            dataset = safe_dataset_name(input_xml)
            row = {
                "dataset": dataset,
                "mode": args.mode,
                "input_xml": str(input_xml),
                "status": f"failed: {type(exc).__name__}: {exc}",
                "input_passages": count_bioc_passages(input_xml),
                "input_annotations": count_bioc_annotations(input_xml),
            }
            rows.append(row)
            print(f"{dataset}: {row['status']}", file=sys.stderr)
            if not args.continue_on_error:
                break

    write_csv(rows, manifest)
    print(f"Wrote run manifest to {manifest}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
