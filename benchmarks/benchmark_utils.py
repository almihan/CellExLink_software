"""Shared utilities for CellExLink benchmark scripts."""
from __future__ import annotations
import csv, json, re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence
from xml.etree import ElementTree as ET

PathLike = str | Path
GENERIC_RELAXED_BLACKLIST = frozenset({"cell", "cells", "cellular", "population", "populations", "line", "lines"})
CL_ID_RE = re.compile(r"CL[:_]?(\d{7})", re.I)
GENERIC_NORMALIZATION_INFON_KEYS = frozenset(
    {
        "cl_id",
        "CL_ID",
        "identifier",
        "identifier_id",
        "identifier_name",
        "ontology_id",
        "norm_concept_id",
        "score",
        "match_source",
        "source",
    }
)

@dataclass(frozen=True, slots=True)
class Mention:
    document_id: str
    passage_id: int
    start: int
    end: int
    text: str = ""
    label: str = "cell_type"
    cl_id: Optional[str] = None
    infons: Mapping[str, str] = field(default_factory=dict)
    @property
    def span_key(self) -> tuple[str, int, int, int]: return (self.document_id, self.passage_id, self.start, self.end)
    @property
    def linked_key(self) -> tuple[str, int, int, int, Optional[str]]: return (*self.span_key, normalize_cl_id(self.cl_id))
    def normalized_text(self) -> str: return normalize_text(self.text)
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self); d["cl_id"] = normalize_cl_id(self.cl_id); return d

@dataclass(slots=True)
class Metrics:
    tp: int; fp: int; fn: int
    @property
    def precision(self) -> float: return self.tp/(self.tp+self.fp) if self.tp+self.fp else 0.0
    @property
    def recall(self) -> float: return self.tp/(self.tp+self.fn) if self.tp+self.fn else 0.0
    @property
    def f1(self) -> float:
        p,r=self.precision,self.recall; return 2*p*r/(p+r) if p+r else 0.0
    def to_row(self, **extra: Any) -> dict[str, Any]:
        row=dict(extra); row.update(tp=self.tp, fp=self.fp, fn=self.fn, precision=self.precision, recall=self.recall, f1=self.f1); return row

def normalize_text(text: str) -> str: return re.sub(r"\s+", " ", (text or "").strip().lower())
def normalize_cl_id(value: Any) -> Optional[str]:
    if value is None: return None
    text=str(value).strip()
    if not text or text.lower() in {"none","nan","null","-","--"}: return None
    m=CL_ID_RE.search(text)
    if m: return f"CL:{m.group(1)}"
    return text.upper() if text.upper().startswith("CL:") else text

def _safe_int(v: Any, default: int=0) -> int:
    try: return int(v)
    except (TypeError, ValueError): return default

def _child_text(e: ET.Element, name: str, default: str="") -> str:
    c=e.find(name); return c.text if c is not None and c.text is not None else default

def _infons(e: ET.Element) -> dict[str,str]: return {i.attrib.get("key",""): i.text or "" for i in e.findall("infon") if i.attrib.get("key")}
def _cl_from_infons(infons: Mapping[str,str]) -> Optional[str]:
    for k in ("cl_id","CL_ID","identifier","identifier_id","ontology_id","norm_concept_id","CellExLink-Sapbert_id_0","CellExLink-SapBERT_id_0"):
        if k in infons and normalize_cl_id(infons[k]): return normalize_cl_id(infons[k])
    for k,v in infons.items():
        lk=k.lower()
        if lk.endswith("_id_0") or lk.endswith("_cl_id_0") or "cell ontology" in lk:
            if normalize_cl_id(v): return normalize_cl_id(v)
    for v in infons.values():
        n=normalize_cl_id(v)
        if n and n.startswith("CL:"): return n
    return None

def read_bioc_mentions(path: PathLike, *, exclude_types: Iterable[str] = ("cell_vague",)) -> list[Mention]:
    path=Path(path)
    if not path.is_file(): raise FileNotFoundError(path)
    excluded={x.lower() for x in exclude_types}; root=ET.parse(path).getroot(); out=[]
    for doc in root.findall(".//document"):
        doc_id=_child_text(doc,"id","")
        for pidx, passage in enumerate(doc.findall("passage")):
            poff=_safe_int(_child_text(passage,"offset","0"),0); ptext=_child_text(passage,"text","")
            for ann in passage.findall("annotation"):
                inf=_infons(ann); label=inf.get("type") or inf.get("entity_type") or inf.get("label") or "cell_type"
                if label.lower() in excluded: continue
                ann_text=_child_text(ann,"text",""); cl_id=_cl_from_infons(inf)
                for loc in ann.findall("location"):
                    start=_safe_int(loc.attrib.get("offset"),-1); length=_safe_int(loc.attrib.get("length"),-1)
                    if start<0 or length<=0: continue
                    end=start+length; ls=start-poff; le=ls+length
                    text=ann_text or (ptext[ls:le] if 0<=ls<=le<=len(ptext) else "")
                    out.append(Mention(doc_id,pidx,start,end,text,label,cl_id,inf))
    return out

def _entity_cl_id(entity: Mapping[str,Any]) -> Optional[str]:
    for key in ("cl_id","identifier","identifier_id","norm_concept_id","ontology_id"):
        n=normalize_cl_id(entity.get(key))
        if n: return n
    inf=entity.get("infons")
    return _cl_from_infons({str(k):str(v) for k,v in inf.items()}) if isinstance(inf, Mapping) else None

def read_jsonl_mentions(path: PathLike) -> list[Mention]:
    out=[]
    with Path(path).open("r",encoding="utf-8") as h:
        for ln,line in enumerate(h,1):
            if not line.strip(): continue
            row=json.loads(line)
            if "entities" in row:
                doc_id=str(row.get("document_id") or row.get("doc_id") or ""); pidx=_safe_int(row.get("passage_id", row.get("passage_index",0)),0); poff=_safe_int(row.get("passage_offset", row.get("offset",0)),0); ptext=str(row.get("text") or "")
                for ent in row.get("entities") or []:
                    if not isinstance(ent, Mapping): continue
                    s=_safe_int(ent.get("start"),-1); e=_safe_int(ent.get("end"),-1)
                    if s<0 or e<=s: continue
                    text=str(ent.get("text") or ptext[s:e])
                    out.append(Mention(doc_id,pidx,poff+s,poff+e,text,str(ent.get("label") or ent.get("type") or "cell_type"),_entity_cl_id(ent),dict(ent.get("infons") or {})))
            else:
                s=_safe_int(row.get("start", row.get("offset",-1)),-1); e=_safe_int(row.get("end"),-1)
                if e<=s and "length" in row: e=s+_safe_int(row.get("length"),0)
                if s<0 or e<=s: raise ValueError(f"Invalid offsets on line {ln} in {path}")
                out.append(Mention(str(row.get("document_id") or row.get("doc_id") or ""),_safe_int(row.get("passage_id", row.get("passage_index",0)),0),s,e,str(row.get("text") or row.get("mention") or ""),str(row.get("label") or row.get("type") or "cell_type"),normalize_cl_id(row.get("cl_id") or row.get("identifier")),dict(row.get("infons") or {})))
    return out

def load_mentions(path: PathLike, *, exclude_types: Iterable[str] = ("cell_vague",)) -> list[Mention]:
    suffix=Path(path).suffix.lower()
    if suffix==".xml": return read_bioc_mentions(path, exclude_types=exclude_types)
    if suffix in {".jsonl",".ndjson"}: return read_jsonl_mentions(path)
    raise ValueError(f"Unsupported file type {suffix}; use .xml or .jsonl")

def parse_named_paths(values: Sequence[str]) -> dict[str,Path]:
    parsed={}
    for v in values:
        if "=" in v:
            name,p=v.split("=",1); parsed[name.strip()]=Path(p)
        else:
            p=Path(v); parsed[p.stem]=p
    return parsed

def pair_named_paths(gold_values: Sequence[str], pred_values: Sequence[str]) -> list[tuple[str,Path,Path]]:
    gold,pred=parse_named_paths(gold_values),parse_named_paths(pred_values)
    if set(gold)!=set(pred): raise ValueError(f"Dataset names differ: gold={sorted(gold)}, pred={sorted(pred)}")
    return [(n,gold[n],pred[n]) for n in sorted(gold)]

def mentions_overlap(a: Mention,b: Mention) -> bool: return a.document_id==b.document_id and a.passage_id==b.passage_id and max(a.start,b.start)<min(a.end,b.end)
def ids_match(a: Mention,b: Mention) -> bool: return bool(normalize_cl_id(a.cl_id) and normalize_cl_id(a.cl_id)==normalize_cl_id(b.cl_id))
def evaluate_mentions(gold: Sequence[Mention], pred: Sequence[Mention], *, criterion="exact", require_cl_id=False, relaxed_blacklist: Iterable[str]=GENERIC_RELAXED_BLACKLIST) -> Metrics:
    mg,mp=set(),set()
    if criterion=="exact":
        lookup={}
        for i,g in enumerate(gold): lookup.setdefault(g.linked_key if require_cl_id else g.span_key,[]).append(i)
        for j,p in enumerate(pred):
            for i in lookup.get(p.linked_key if require_cl_id else p.span_key,[]):
                if i not in mg: mg.add(i); mp.add(j); break
    elif criterion=="relaxed":
        blacklist={normalize_text(x) for x in relaxed_blacklist}
        for j,p in enumerate(pred):
            if p.normalized_text() in blacklist: continue
            best_i=None; best_ov=-1
            for i,g in enumerate(gold):
                if i in mg or not mentions_overlap(g,p): continue
                if require_cl_id and not ids_match(g,p): continue
                ov=min(g.end,p.end)-max(g.start,p.start)
                if ov>best_ov: best_i,best_ov=i,ov
            if best_i is not None: mg.add(best_i); mp.add(j)
    else: raise ValueError("criterion must be exact or relaxed")
    return Metrics(len(mg),len(pred)-len(mp),len(gold)-len(mg))

def write_csv(rows: Iterable[Mapping[str,Any]], path: PathLike) -> Path:
    rows=list(rows); path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with path.open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(rows)
    return path

def read_csv(path: PathLike) -> list[dict[str,str]]:
    with Path(path).open("r",newline="",encoding="utf-8") as h: return list(csv.DictReader(h))
def format_float(value: Any, digits=3) -> str:
    try: return f"{float(value):.{digits}f}"
    except (TypeError,ValueError): return str(value)
def count_bioc_passages(path: PathLike) -> int: return len(ET.parse(path).getroot().findall(".//passage"))
def count_bioc_annotations(path: PathLike) -> int: return len(ET.parse(path).getroot().findall(".//annotation"))

def _should_strip_normalization_infon(key: str) -> bool:
    if key in GENERIC_NORMALIZATION_INFON_KEYS: return True
    suffixes=(
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
    return any(key.endswith(suffix) for suffix in suffixes)

def strip_normalization_infons(input_xml: PathLike, output_xml: PathLike) -> Path:
    src=Path(input_xml); dst=Path(output_xml)
    tree=ET.parse(src); root=tree.getroot()
    for ann in root.findall(".//annotation"):
        for infon in list(ann.findall("infon")):
            key=infon.attrib.get("key","")
            if _should_strip_normalization_infon(key):
                ann.remove(infon)
    dst.parent.mkdir(parents=True,exist_ok=True)
    tree.write(dst, encoding="utf-8", xml_declaration=True)
    return dst
