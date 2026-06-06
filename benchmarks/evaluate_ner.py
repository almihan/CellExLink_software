#!/usr/bin/env python
"""Evaluate cell-type NER predictions with exact and relaxed span matching."""
from __future__ import annotations
import argparse
from pathlib import Path
from benchmark_utils import GENERIC_RELAXED_BLACKLIST, evaluate_mentions, load_mentions, pair_named_paths, write_csv

def build_parser():
    p=argparse.ArgumentParser(description="Evaluate CellExLink-style NER predictions.")
    p.add_argument("--gold", action="append", required=True, help="Gold file as DATASET=path or path. Repeatable.")
    p.add_argument("--pred", action="append", required=True, help="Prediction file as DATASET=path or path. Repeatable.")
    p.add_argument("--system", default="CellExLink"); p.add_argument("--output-csv", default="benchmark_outputs/ner_results.csv")
    p.add_argument("--exclude-type", action="append", default=["cell_vague"]); p.add_argument("--relaxed-blacklist", action="append", default=sorted(GENERIC_RELAXED_BLACKLIST)); p.add_argument("--quiet", action="store_true")
    return p

def main():
    a=build_parser().parse_args(); rows=[]
    for dataset,gold_path,pred_path in pair_named_paths(a.gold,a.pred):
        gold=load_mentions(gold_path, exclude_types=a.exclude_type); pred=load_mentions(pred_path, exclude_types=a.exclude_type)
        for criterion in ("exact","relaxed"):
            m=evaluate_mentions(gold,pred,criterion=criterion,relaxed_blacklist=a.relaxed_blacklist)
            rows.append(m.to_row(system=a.system,dataset=dataset,task="NER",criterion=f"{criterion}_span",gold_file=str(gold_path),pred_file=str(pred_path)))
            if not a.quiet: print(f"{a.system}\t{dataset}\t{criterion}\tP={m.precision:.3f}\tR={m.recall:.3f}\tF1={m.f1:.3f}")
    write_csv(rows,a.output_csv)
    if not a.quiet: print(f"Wrote NER results to {Path(a.output_csv)}")
    return 0
if __name__=="__main__": raise SystemExit(main())
