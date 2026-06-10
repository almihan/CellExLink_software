#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import uuid
from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET


BASELINE_DIR = Path(__file__).resolve().parent
BENCHMARKS_DIR = BASELINE_DIR.parents[1]
DATA_DIR = BENCHMARKS_DIR / "data" / "evaluation"
OUTPUT_DIR = BASELINE_DIR / "model_outputs"

BERN2_ROOT = BASELINE_DIR / "BERN2"
MODEL_NAME_OR_PATH = BASELINE_DIR / "models" / "dmis-lab" / "bern2-ner"
DICTIONARY_PATH = BERN2_ROOT / "resources" / "normalization" / "dictionary" / "dict_CellType_20210810.txt"

DATASET = os.environ.get("DATASET", "CRAFT")
SPLIT = os.environ.get("SPLIT", "test")
INPUT_XML = Path(os.environ.get("INPUT_XML", str(DATA_DIR / DATASET / f"{SPLIT}.xml")))
NER_OUTPUT_XML = OUTPUT_DIR / f"{DATASET}_{SPLIT}.bern2.ner.xml"
NORMALIZED_OUTPUT_XML = OUTPUT_DIR / f"{DATASET}_{SPLIT}.bern2.normalized.xml"

MAX_SEQ_LENGTH = 128
SEED = 42
NO_CUDA = False
CHUNK_CHARS = 2500
BATCH_DOCS = 32
TOPN = 10
MODEL_NAME = "BERN2"

Mention = Dict[str, object]


def local_name(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def segment_text(text: str, max_chars: int = CHUNK_CHARS) -> List[Tuple[int, str]]:
    if len(text) <= max_chars:
        return [(0, text)]

    pieces: List[Tuple[int, int]] = []
    start = 0
    for idx, ch in enumerate(text):
        if ch in ".!?\n":
            end = idx + 1
            if end > start:
                pieces.append((start, end))
            start = end
    if start < len(text):
        pieces.append((start, len(text)))

    chunks: List[Tuple[int, str]] = []
    chunk_start: Optional[int] = None
    chunk_end: Optional[int] = None
    for piece_start, piece_end in pieces:
        if piece_end - piece_start > max_chars:
            cursor = piece_start
            while cursor < piece_end:
                end = min(cursor + max_chars, piece_end)
                if end < piece_end:
                    ws = text.rfind(" ", cursor, end)
                    if ws > cursor + max_chars // 2:
                        end = ws + 1
                chunks.append((cursor, text[cursor:end]))
                cursor = end
            continue
        if chunk_start is None:
            chunk_start, chunk_end = piece_start, piece_end
        elif piece_end - chunk_start <= max_chars:
            chunk_end = piece_end
        else:
            chunks.append((chunk_start, text[chunk_start:chunk_end]))
            chunk_start, chunk_end = piece_start, piece_end

    if chunk_start is not None and chunk_end is not None:
        chunks.append((chunk_start, text[chunk_start:chunk_end]))
    return [(offset, chunk) for offset, chunk in chunks if chunk.strip()]


def load_mtner():
    sys.path.insert(0, str(BERN2_ROOT.resolve()))
    sys.path.insert(0, str((BERN2_ROOT / "multi_ner").resolve()))
    from main import MTNER

    return MTNER(
        Namespace(
            seed=SEED,
            model_name_or_path=str(MODEL_NAME_OR_PATH),
            max_seq_length=MAX_SEQ_LENGTH,
            no_cuda=NO_CUDA,
        )
    )


def load_celltype_normalizer():
    if not DICTIONARY_PATH.is_file():
        raise FileNotFoundError(f"BERN2 cell-type dictionary not found: {DICTIONARY_PATH}")

    sys.path.insert(0, str(BERN2_ROOT.resolve()))
    from normalizers.celltype_normalizer import CellTypeNormalizer

    return CellTypeNormalizer(str(DICTIONARY_PATH))


def extract_cell_types(mtner_doc: Dict[str, object], text: str) -> List[Mention]:
    entities = mtner_doc.get("entities", {})
    if not isinstance(entities, dict):
        return []

    mentions: List[Mention] = []
    for entity in entities.get("cell_type", []):
        if not isinstance(entity, dict):
            continue
        try:
            start = int(entity.get("start"))
            end = int(entity.get("end")) + 1
        except Exception:
            continue
        if 0 <= start < end <= len(text):
            mentions.append({"spans": ((start, end),), "text": text[start:end]})
    return mentions


def run_mtner_batch(mtner, batch: List[Dict[str, object]], devnull) -> List[List[Mention]]:
    mtner_input = [
        {"pmid": item["pmid"], "title": "", "abstract": item["text"], "entities": {}}
        for item in batch
    ]

    with redirect_stdout(devnull), redirect_stderr(devnull):
        mtner_docs = mtner.recognize(input_dl=mtner_input, base_name=uuid.uuid4().hex)

    if not isinstance(mtner_docs, list) or len(mtner_docs) != len(batch):
        mtner_docs = []
        for one_doc in mtner_input:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                one_result = mtner.recognize(input_dl=[one_doc], base_name=uuid.uuid4().hex)
            mtner_docs.append(one_result[0] if isinstance(one_result, list) and one_result else one_doc)

    return [extract_cell_types(doc, str(item["text"])) for item, doc in zip(batch, mtner_docs)]


def predict_cell_types(mtner, passage_text: str, devnull) -> List[Mention]:
    predictions: List[Mention] = []
    batch: List[Dict[str, object]] = []

    def flush_batch() -> None:
        nonlocal batch
        if not batch:
            return
        for item, mentions in zip(batch, run_mtner_batch(mtner, batch, devnull)):
            chunk_offset = int(item["offset"])
            for mention in mentions:
                start, end = mention["spans"][0]
                predictions.append(
                    {
                        "spans": ((chunk_offset + int(start), chunk_offset + int(end)),),
                        "text": mention["text"],
                    }
                )
        batch = []

    for chunk_offset, chunk_text in segment_text(passage_text):
        batch.append({"pmid": uuid.uuid4().hex, "offset": chunk_offset, "text": chunk_text})
        if len(batch) >= BATCH_DOCS:
            flush_batch()
    flush_batch()

    dedup: Dict[Tuple[Tuple[int, int], str], Mention] = {}
    for mention in predictions:
        dedup[(tuple(mention["spans"][0]), str(mention["text"]))] = mention
    return list(dedup.values())


def next_annotation_id(document: ET.Element) -> int:
    max_id = -1
    for annotation in document.iterfind(".//annotation"):
        raw_id = annotation.get("id")
        if raw_id is None:
            continue
        try:
            max_id = max(max_id, int(raw_id))
        except ValueError:
            continue
    return max_id + 1


def clear_passage_predictions(passage: ET.Element) -> None:
    for child in list(passage):
        if local_name(child.tag) in {"annotation", "relation"}:
            passage.remove(child)


def add_annotation(passage: ET.Element, annotation_id: int, offset: int, length: int, text: str) -> None:
    annotation = ET.SubElement(passage, "annotation", {"id": str(annotation_id)})
    infon = ET.SubElement(annotation, "infon", {"key": "type"})
    infon.text = "cell_type"
    ET.SubElement(annotation, "location", {"offset": str(offset), "length": str(length)})
    text_node = ET.SubElement(annotation, "text")
    text_node.text = text


def run_ner(input_xml: Path, output_xml: Path) -> int:
    mtner = load_mtner()
    tree = ET.parse(input_xml)
    root = tree.getroot()
    annotations_added = 0

    with open(os.devnull, "w") as devnull:
        for document in root.findall("document"):
            next_id = next_annotation_id(document)
            for passage in document.findall("passage"):
                clear_passage_predictions(passage)
                text_node = passage.find("text")
                offset_node = passage.find("offset")
                if text_node is None or text_node.text is None or offset_node is None or offset_node.text is None:
                    continue

                passage_text = text_node.text
                passage_offset = int(offset_node.text)
                for mention in predict_cell_types(mtner, passage_text, devnull):
                    start, end = mention["spans"][0]
                    add_annotation(
                        passage=passage,
                        annotation_id=next_id,
                        offset=passage_offset + int(start),
                        length=int(end) - int(start),
                        text=str(mention["text"]),
                    )
                    next_id += 1
                    annotations_added += 1

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    return annotations_added


def normalize_with_bern2(normalizer, mention_text: str) -> List[str]:
    normalized = normalizer.normalize([mention_text])
    if not normalized or not isinstance(normalized, list):
        return []

    identifiers = normalized[0]
    if identifiers is None:
        return []
    if isinstance(identifiers, str):
        identifiers = [identifiers]
    if not isinstance(identifiers, list):
        return []

    cleaned: List[str] = []
    for identifier in identifiers:
        if not isinstance(identifier, str):
            continue
        identifier = identifier.strip()
        if identifier.startswith("CL_"):
            identifier = identifier.replace("CL_", "CL:", 1)
        if identifier and identifier.lower() != "cui-less" and identifier not in cleaned:
            cleaned.append(identifier)
    return cleaned


def run_normalization(input_xml: Path, output_xml: Path) -> Tuple[int, int, int]:
    import bioc

    normalizer = load_celltype_normalizer()
    with input_xml.open("r", encoding="utf-8") as readfp:
        collection = bioc.load(readfp)

    mention_cache: Dict[str, List[str]] = {}
    processed = matched = missing = 0

    for document in collection.documents:
        for passage in document.passages:
            if not passage.infons.get("annotatable", True):
                continue

            for annotation in passage.annotations:
                if annotation.infons.get("type") == "cell_vague":
                    continue

                processed += 1
                mention_text = (annotation.text or "").strip()
                if not mention_text:
                    missing += 1
                    continue

                if mention_text not in mention_cache:
                    mention_cache[mention_text] = normalize_with_bern2(normalizer, mention_text)

                normalized_ids = mention_cache[mention_text]
                if not normalized_ids:
                    missing += 1
                    continue

                matched += 1
                for rank, identifier in enumerate(normalized_ids[:TOPN]):
                    annotation.infons[f"{MODEL_NAME}_id_{rank}"] = identifier
                    if rank == 0:
                        annotation.infons[f"{MODEL_NAME}_identifier_name_{rank}"] = mention_text
                        annotation.infons[f"{MODEL_NAME}_identifier_score_{rank}"] = "1.0"

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    with output_xml.open("w", encoding="utf-8") as writefp:
        bioc.dump(collection, writefp)

    return processed, matched, missing


def main() -> None:
    print(f"Input XML: {INPUT_XML}")
    print(f"BERN2 root: {BERN2_ROOT}")
    print(f"BERN2 model: {MODEL_NAME_OR_PATH}")
    print(f"Output NER XML: {NER_OUTPUT_XML}")
    print(f"Output normalized XML: {NORMALIZED_OUTPUT_XML}")

    added = run_ner(INPUT_XML, NER_OUTPUT_XML)
    processed, matched, missing = run_normalization(NER_OUTPUT_XML, NORMALIZED_OUTPUT_XML)

    print(f"NER annotations added: {added}")
    print(f"Normalization processed: {processed}, matched: {matched}, missing: {missing}")


if __name__ == "__main__":
    main()
