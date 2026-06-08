#!/usr/bin/env python
"""Evaluate strict CellExLink end-to-end extraction.

A prediction is counted as correct only when the passage key, entity type,
exact BioC location tuple, and Cell Ontology identifier all match the reference.
Use this for Table 7-style strict end-to-end results.
"""

from __future__ import annotations

try:
    from .eval_original_compatible import (
        build_common_parser,
        evaluate_file_pair,
        pair_named_paths,
        print_rows,
        write_csv,
    )
except ImportError:  # Allows: python benchmarks/evaluate_*.py
    from eval_original_compatible import (
        build_common_parser,
        evaluate_file_pair,
        pair_named_paths,
        print_rows,
        write_csv,
    )


def main() -> int:
    parser = build_common_parser("Evaluate strict end-to-end CellExLink output.")
    args = parser.parse_args()

    rows = []
    for dataset_name, gold_path, pred_path in pair_named_paths(args.gold, args.pred):
        rows.extend(
            evaluate_file_pair(
                dataset_name=dataset_name,
                gold_file=gold_path,
                pred_file=pred_path,
                dataset_style=args.dataset_style,
                score_mode="end_to_end",
                model_names=args.model_names,
                topk=args.topk,
                score_threshold=args.threshold,
            )
        )

    write_csv(rows, args.output_csv)
    if not args.quiet:
        print_rows(rows)
        print(f"Wrote strict end-to-end results to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
