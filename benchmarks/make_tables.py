#!/usr/bin/env python
"""Generate Markdown benchmark tables from result CSV files."""
from __future__ import annotations
import argparse
from pathlib import Path
from benchmark_utils import format_float, read_csv

def find_result_file(results_dir: Path, preferred: str, fallback: str):
    for name in (preferred, fallback):
        p=results_dir/name
        if p.is_file(): return p
    return None

def md(headers, rows):
    lines=["| "+" | ".join(headers)+" |", "| "+" | ".join(["---"]*len(headers))+" |"]
    for row in rows: lines.append("| "+" | ".join(row)+" |")
    return "\n".join(lines)+"\n"

def write(path: Path, title: str, table: str):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(f"# {title}\n\n{table}\n", encoding="utf-8")

def table2(path, out):
    rows=read_csv(path); datasets=["CellLink","CRAFT","BioID","AnatEM","JNLPBA","Macro avg."]; systems=[]; lookup={}
    for r in rows:
        if r["system"] not in systems: systems.append(r["system"])
        lookup[(r["system"],r["dataset"])] = r
    trs=[]
    for s in systems:
        row=[s]
        for d in datasets:
            item=lookup.get((s,d),{}); row += [format_float(item.get("exact_f1","")), format_float(item.get("relaxed_f1",""))]
        trs.append(row)
    h=["System"]
    for d in datasets: h += [f"{d} E", f"{d} R"]
    write(out/"table2_ner_results.md", "Table 2. Cell-type recognition", md(h,trs))

def table4(path,out):
    trs=[[r.get("task",""),r.get("model",""),r.get("size_m",""),format_float(r.get("inference_cost","")),r.get("unit","")] for r in read_csv(path)]
    write(out/"table4_runtime_results.md", "Table 4. Model size and inference cost", md(["Task","Model","Size (M)","Inference cost","Unit"], trs))

def table5(path,out):
    trs=[[r.get("dataset",""),r.get("condition",""),r.get("entity_group",""),format_float(r.get("BERN2","")),format_float(r.get("SciSpacy",r.get("ScispaCy",""))),format_float(r.get("CellExLink",""))] for r in read_csv(path)]
    write(out/"table5_gold_span_normalization_results.md", "Table 5. Gold-span normalization", md(["Dataset","Condition","Entity group","BERN2","SciSpacy","CellExLink"], trs))

def table6(path,out):
    trs=[[r.get("group",""),r.get("linker",""),format_float(r.get("CellLink_exact","")),format_float(r.get("CellLink_all","")),format_float(r.get("CRAFT","")),format_float(r.get("BioID",""))] for r in read_csv(path)]
    write(out/"table6_linker_selection_results.md", "Table 6. Linker selection", md(["Group","Linker","CellLink exact","CellLink all","CRAFT","BioID"], trs))

def table7(path,out):
    rows=read_csv(path); datasets=["CellLink","CRAFT","BioID"]; systems=[]; lookup={}
    for r in rows:
        if r["system"] not in systems: systems.append(r["system"])
        lookup[(r["system"],r["dataset"])] = r
    trs=[]
    for s in systems:
        row=[s]
        for d in datasets:
            item=lookup.get((s,d),{}); row += [format_float(item.get("precision","")), format_float(item.get("recall","")), format_float(item.get("f1",""))]
        trs.append(row)
    h=["System"]
    for d in datasets: h += [f"{d} P", f"{d} R", f"{d} F1"]
    write(out/"table7_end_to_end_results.md", "Table 7. Strict end-to-end extraction", md(h,trs))

def build_parser():
    p=argparse.ArgumentParser(description="Generate Markdown benchmark tables from CSV files.")
    p.add_argument("--results-dir", default="benchmarks/results"); p.add_argument("--output-dir", default="benchmark_outputs/tables")
    return p

def main():
    a=build_parser().parse_args(); rd=Path(a.results_dir); out=Path(a.output_dir)
    jobs=[("table2_ner_results.csv","reference_table2_ner_results.csv",table2),("table4_runtime_results.csv","reference_table4_runtime_results.csv",table4),("table5_gold_span_normalization_results.csv","reference_table5_gold_span_normalization_results.csv",table5),("table6_linker_selection_results.csv","reference_table6_linker_selection_results.csv",table6),("table7_end_to_end_results.csv","reference_table7_end_to_end_results.csv",table7)]
    generated=[]
    for preferred, fallback, maker in jobs:
        p=find_result_file(rd,preferred,fallback)
        if p is None: print(f"Skipping missing result file: {preferred} or {fallback}"); continue
        maker(p,out); generated.append(p.name)
    print(f"Generated tables in {out}: {', '.join(generated) if generated else 'none'}"); return 0
if __name__=="__main__": raise SystemExit(main())
