#!/usr/bin/env python3
"""Convenience runner for the optional ScispaCy baseline.

Examples
--------
NER only:

    python benchmarks/baselines/scispacy/run_scispacy.py \
      --input benchmarks/data/evaluation/JNLPBA/input.xml \
      --output-dir benchmark_outputs/scispacy \
      --mode ner

Full NER + NEN:

    python benchmarks/baselines/scispacy/run_scispacy.py \
      --input benchmarks/data/evaluation/CellLink/input.xml \
      --output-dir benchmark_outputs/scispacy \
      --mode full
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from benchmarks.benchmark_utils import strip_normalization_infons


BASELINE_DIR = Path(__file__).resolve().parent
PREDICT_NER = BASELINE_DIR / "predict_ner.py"
PREDICT_NEN = BASELINE_DIR / "predict_nen.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run optional ScispaCy benchmark baseline.")
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=Path,
        help="Input BioC XML. May be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for ScispaCy baseline outputs.",
    )
    parser.add_argument(
        "--mode",
        choices=("ner", "normalize", "full"),
        required=True,
        help="Run NER only, normalization only, or full NER+normalization.",
    )
    parser.add_argument("--model-name", default="en_ner_craft_md", help="ScispaCy NER model.")
    parser.add_argument("--batch-size", type=int, default=64, help="ScispaCy NER batch size.")
    parser.add_argument(
        "--offset-mode",
        choices=("char", "bioc_bytes"),
        default="char",
        help="BioC offset mode.",
    )
    parser.add_argument("--ontology-prefix", default="cl", help="PyOBO ontology prefix.")
    parser.add_argument("--topn", type=int, default=10, help="Number of linker candidates.")
    parser.add_argument("--score-threshold", type=float, default=0.0, help="Linker score threshold.")
    parser.add_argument(
        "--disable-abbreviations",
        action="store_true",
        help="Disable ScispaCy abbreviation expansion during normalization.",
    )
    parser.add_argument(
        "--strip-input-id-infons",
        action="store_true",
        help=(
            "For mode=normalize, create a temporary BioC copy with gold/system "
            "normalization infons removed while preserving mention spans."
        ),
    )
    parser.add_argument(
        "--fail-if-input-has-annotations",
        action="store_true",
        help=(
            "For mode=ner/full, fail when input XML contains annotations. "
            "Use this with text-only input.xml for strict no-leakage benchmark runs."
        ),
    )
    parser.add_argument("--manifest", type=Path, default=None, help="Optional output CSV manifest.")
    return parser.parse_args()


def dataset_stem(path: Path) -> str:
    dataset = path.parent.name
    stem = path.stem
    return f"{dataset}_{stem}"


def count_bioc_annotations(path: Path) -> int:
    root = ET.parse(path).getroot()
    return len(root.findall(".//annotation"))


def check_no_leakage_input(path: Path, *, fail: bool) -> None:
    count = count_bioc_annotations(path)
    if count == 0:
        return

    message = (
        f"{path} contains {count} BioC annotations. ScispaCy NER will clear existing "
        "annotations before writing predictions, but for a strict no-leakage benchmark "
        "use text-only input.xml and keep gold annotations in gold.xml."
    )

    if fail:
        raise RuntimeError(message)

    print(f"WARNING: {message}", file=sys.stderr)


def run_command(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def write_manifest(rows: list[dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []

    try:
        for input_xml in args.input:
            input_xml = input_xml.resolve()
            if not input_xml.is_file():
                raise FileNotFoundError(f"Input XML does not exist: {input_xml}")

            prefix = dataset_stem(input_xml)
            ner_xml = args.output_dir / f"{prefix}.scispacy.ner.xml"
            normalized_xml = args.output_dir / f"{prefix}.scispacy.normalized.xml"

            start_time = time.perf_counter()

            if args.mode in {"ner", "full"}:
                check_no_leakage_input(
                    input_xml,
                    fail=args.fail_if_input_has_annotations,
                )
                run_command(
                    [
                        sys.executable,
                        str(PREDICT_NER),
                        "--input-xml",
                        str(input_xml),
                        "--output-xml",
                        str(ner_xml),
                        "--model-name",
                        args.model_name,
                        "--batch-size",
                        str(args.batch_size),
                        "--offset-mode",
                        args.offset_mode,
                    ]
                )

            if args.mode == "normalize":
                normalization_input = input_xml
                if args.strip_input_id_infons:
                    stripped_xml = args.output_dir / "work" / f"{prefix}.normalize_input.stripped.xml"
                    stripped_xml.parent.mkdir(parents=True, exist_ok=True)
                    normalization_input = strip_normalization_infons(input_xml, stripped_xml)
            elif args.mode == "full":
                normalization_input = ner_xml
            else:
                normalization_input = None

            if normalization_input is not None:
                command = [
                    sys.executable,
                    str(PREDICT_NEN),
                    "--input-xml",
                    str(normalization_input),
                    "--output-xml",
                    str(normalized_xml),
                    "--ontology-prefix",
                    args.ontology_prefix,
                    "--topn",
                    str(args.topn),
                    "--score-threshold",
                    str(args.score_threshold),
                ]
                if args.disable_abbreviations:
                    command.append("--disable-abbreviations")
                run_command(command)

            elapsed = time.perf_counter() - start_time

            rows.append(
                {
                    "system": "ScispaCy",
                    "mode": args.mode,
                    "input_xml": str(input_xml),
                    "strip_input_id_infons": args.strip_input_id_infons,
                    "ner_xml": str(ner_xml) if ner_xml.exists() else "",
                    "normalized_xml": str(normalized_xml) if normalized_xml.exists() else "",
                    "runtime_seconds": f"{elapsed:.6f}",
                }
            )

    except Exception as exc:
        print(f"ScispaCy baseline failed: {exc}", file=sys.stderr)
        return 1

    manifest = args.manifest or (args.output_dir / "scispacy_run_manifest.csv")
    write_manifest(rows, manifest)
    print(f"Wrote manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
