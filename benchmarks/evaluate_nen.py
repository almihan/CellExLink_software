#!/usr/bin/env python
"""Evaluate CellExLink gold-span Cell Ontology normalization.

This is the SoftwareX-minimal wrapper around the original-compatible NEN
metric.  Use this for Table 5-style results.

Important: the prediction XML must preserve the gold ``identifier`` infons from
the input BioC XML and add predicted fields such as
``CellExLink-Sapbert_id_0``.  Do not run this on a prediction XML where gold
``identifier`` infons were stripped.
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
    parser = build_common_parser("Evaluate gold-span Cell Ontology normalization.")
    args = parser.parse_args()

    rows = []
    for dataset_name, gold_path, pred_path in pair_named_paths(args.gold, args.pred):
        rows.extend(
            evaluate_file_pair(
                dataset_name=dataset_name,
                gold_file=gold_path,
                pred_file=pred_path,
                dataset_style=args.dataset_style,
                score_mode="gold_mention_normalize",
                model_names=args.model_names,
                topk=args.topk,
                score_threshold=args.threshold,
            )
        )

    write_csv(rows, args.output_csv)
    if not args.quiet:
        print_rows(rows)
        print(f"Wrote NEN results to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
