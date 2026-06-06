#!/usr/bin/env python
"""Run CellExLink predictions for benchmark BioC XML files."""
from __future__ import annotations
import argparse, time
from pathlib import Path
from typing import Iterable
from benchmark_utils import count_bioc_annotations, count_bioc_passages, strip_normalization_infons, write_csv

def discover_xml_files(inputs: Iterable[str], pattern: str) -> list[Path]:
    files=[]
    for item in inputs:
        path=Path(item)
        if path.is_dir(): files.extend(sorted(path.glob(pattern)))
        elif path.is_file(): files.append(path)
        else: raise FileNotFoundError(f"Input does not exist: {path}")
    return files

def safe_dataset_name(path: Path) -> str:
    parent, stem = path.parent.name, path.stem
    return f"{parent}_{stem}" if parent and parent.lower() not in {".","data","evaluation","test","train","validation"} else stem

def build_parser():
    p=argparse.ArgumentParser(description="Run CellExLink on benchmark BioC XML files.")
    p.add_argument("--input", action="append", required=True, help="Input BioC XML file or directory. Repeatable.")
    p.add_argument("--glob", default="*.xml"); p.add_argument("--output-dir", default="benchmark_outputs/cellexlink")
    p.add_argument("--mode", choices=["full","ner","normalize"], default="full")
    p.add_argument("--ner-model", default="almire/CellExLink-bioformer16L"); p.add_argument("--nen-model", default="almire/CellExLink-Sapbert")
    p.add_argument("--ontology-path", default=None); p.add_argument("--abbreviations-path", default=None); p.add_argument("--batch-size", type=int, default=16); p.add_argument("--fp16", action="store_true"); p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--strip-input-id-infons", action="store_true", help="For mode=normalize, create a temporary BioC copy with gold/system normalization infons removed while preserving mention spans.")
    p.add_argument("--skip-existing", action="store_true"); p.add_argument("--manifest", default=None)
    return p

def main():
    a=build_parser().parse_args()
    from cellexlink import CellExLinkPipeline
    inputs=discover_xml_files(a.input,a.glob)
    if not inputs: raise FileNotFoundError("No input XML files were found.")
    out=Path(a.output_dir); out.mkdir(parents=True, exist_ok=True); manifest=Path(a.manifest) if a.manifest else out/"run_manifest.csv"
    pipe=CellExLinkPipeline.from_pretrained(ner_model=a.ner_model, nen_model=a.nen_model, ontology_path=a.ontology_path, abbreviations_path=a.abbreviations_path, output_dir=out/"work", batch_size=a.batch_size, fp16=a.fp16, trust_remote_code=a.trust_remote_code)
    rows=[]
    for inp in inputs:
        ds=safe_dataset_name(inp); ner_xml=out/f"{ds}.ner.xml"; norm_xml=out/f"{ds}.normalized.xml"; final=norm_xml if a.mode in {"full","normalize"} else ner_xml
        if a.skip_existing and final.exists(): status="skipped_existing"; elapsed=0.0
        else:
            status="ok"; start=time.perf_counter()
            try:
                if a.mode=="full": pipe.extract_bioc(input_xml=inp, output_xml=norm_xml, ner_output_xml=ner_xml, output_dir=out/"work"/ds)
                elif a.mode=="ner": pipe.recognize_bioc(input_xml=inp, output_xml=ner_xml, output_dir=out/"work"/ds/"ner")
                else:
                    norm_input=inp
                    if a.strip_input_id_infons:
                        stripped_xml=out/"work"/ds/"normalize_input.stripped.xml"
                        stripped_xml.parent.mkdir(parents=True, exist_ok=True)
                        norm_input=strip_normalization_infons(inp, stripped_xml)
                    pipe.normalize_bioc(input_xml=norm_input, output_xml=norm_xml)
            except Exception as exc: status=f"failed: {type(exc).__name__}: {exc}"
            elapsed=time.perf_counter()-start
        rows.append({"dataset":ds,"mode":a.mode,"input_xml":str(inp),"strip_input_id_infons":a.strip_input_id_infons,"ner_xml":str(ner_xml) if ner_xml.exists() else "","normalized_xml":str(norm_xml) if norm_xml.exists() else "","final_output":str(final),"status":status,"elapsed_seconds":elapsed,"input_passages":count_bioc_passages(inp),"input_annotations":count_bioc_annotations(inp),"output_annotations":count_bioc_annotations(final) if final.exists() else ""})
        print(f"{ds}: {status} in {elapsed:.2f}s -> {final}")
    write_csv(rows,manifest); print(f"Wrote run manifest to {manifest}"); return 0
if __name__=="__main__": raise SystemExit(main())
