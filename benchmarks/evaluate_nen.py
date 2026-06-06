#!/usr/bin/env python
"""Evaluate Cell Ontology normalization on gold-standard entity spans."""
from __future__ import annotations
import argparse
from pathlib import Path
from benchmark_utils import evaluate_mentions, load_mentions, pair_named_paths, write_csv

def build_parser():
    p=argparse.ArgumentParser(description="Evaluate CellExLink-style Cell Ontology normalization.")
    p.add_argument("--gold", action="append", required=True); p.add_argument("--pred", action="append", required=True)
    p.add_argument("--system", default="CellExLink"); p.add_argument("--criterion", choices=["exact","relaxed"], default="exact"); p.add_argument("--output-csv", default="benchmark_outputs/nen_results.csv")
    p.add_argument("--exclude-type", action="append", default=["cell_vague"]); p.add_argument("--quiet", action="store_true")
    return p

def main():
    a=build_parser().parse_args(); rows=[]
    for dataset,gold_path,pred_path in pair_named_paths(a.gold,a.pred):
        gold=load_mentions(gold_path, exclude_types=a.exclude_type); pred=load_mentions(pred_path, exclude_types=a.exclude_type)
        m=evaluate_mentions(gold,pred,criterion=a.criterion,require_cl_id=True)
        rows.append(m.to_row(system=a.system,dataset=dataset,task="NEN",criterion=f"{a.criterion}_span_plus_cl_id",gold_file=str(gold_path),pred_file=str(pred_path)))
        if not a.quiet: print(f"{a.system}\t{dataset}\t{a.criterion}+CL\tP={m.precision:.3f}\tR={m.recall:.3f}\tF1={m.f1:.3f}")
    write_csv(rows,a.output_csv)
    if not a.quiet: print(f"Wrote NEN results to {Path(a.output_csv)}")
    return 0
if __name__=="__main__": raise SystemExit(main())
