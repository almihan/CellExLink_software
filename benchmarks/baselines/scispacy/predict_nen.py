#!/usr/bin/env python3
"""Run ScispaCy/PyOBO Cell Ontology linking on BioC annotations.

This script expects BioC XML that already contains mention annotations. It adds
ScispaCy-style normalization infons:

    scispacy_id_0
    scispacy_identifier_name_0
    scispacy_identifier_score_0

Use this for normalization-only or end-to-end ScispaCy baseline evaluation.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import bioc


MODEL_NAME = "scispacy"
DEFAULT_ONTOLOGY_PREFIX = "cl"
DEFAULT_TOPN = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ScispaCy/PyOBO entity linking on BioC XML annotations."
    )
    parser.add_argument("--input-xml", type=Path, required=True, help="Input BioC XML with annotations.")
    parser.add_argument("--output-xml", type=Path, required=True, help="Output linked BioC XML.")
    parser.add_argument(
        "--ontology-prefix",
        default=DEFAULT_ONTOLOGY_PREFIX,
        help=f"PyOBO ontology prefix. Default: {DEFAULT_ONTOLOGY_PREFIX}",
    )
    parser.add_argument("--topn", type=int, default=DEFAULT_TOPN, help="Maximum candidates to keep.")
    parser.add_argument("--score-threshold", type=float, default=0.0, help="Minimum linker score.")
    parser.add_argument(
        "--disable-abbreviations",
        action="store_true",
        help="Disable ScispaCy abbreviation expansion from passage text.",
    )
    parser.add_argument("--debug", action="store_true", help="Print debug information.")
    parser.add_argument(
        "--debug-max-print",
        type=int,
        default=50,
        help="Maximum number of detailed annotation debug blocks.",
    )
    return parser.parse_args()


def debug_print(message: str, enabled: bool) -> None:
    if enabled:
        print(message, flush=True)


def normalize_identifier_case(identifier: str) -> str:
    identifier = (identifier or "").strip()
    return re.sub(r"^cl[_:]", "CL:", identifier, flags=re.IGNORECASE)


def load_celltype_linker(ontology_prefix: str, topn: int, debug: bool):
    try:
        import pyobo
        from scispacy.linking import EntityLinker
    except ImportError as exc:
        raise RuntimeError(
            "pyobo and scispacy are required for this optional baseline. "
            "Create the ScispaCy baseline environment first."
        ) from exc

    debug_print(f"[DEBUG] Loading PyOBO ontology prefix: {ontology_prefix}", debug)

    try:
        return pyobo.get_scispacy_entity_linker(
            ontology_prefix,
            filter_for_definitions=False,
            max_entities_per_mention=topn,
        )
    except Exception as first_exc:
        print(
            f"High-level PyOBO linker construction failed for {ontology_prefix!r}: {first_exc!r}",
            file=sys.stderr,
        )

    try:
        kb = pyobo.get_scispacy_knowledgebase(ontology_prefix)
        return EntityLinker.from_kb(
            kb,
            filter_for_definitions=False,
            max_entities_per_mention=topn,
        )
    except Exception as second_exc:
        raise RuntimeError(
            f"Could not load PyOBO/scispaCy linker for ontology {ontology_prefix!r}."
        ) from second_exc


def build_linking_nlp():
    import spacy

    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns([{"label": "MENTION", "pattern": [{"TEXT": {"REGEX": ".+"}}]}])
    return nlp


def build_abbreviation_nlp():
    import spacy

    try:
        import scispacy.abbreviation  # noqa: F401
    except ImportError:
        print(
            "Warning: scispacy.abbreviation is unavailable; continuing without abbreviation expansion.",
            file=sys.stderr,
        )
        return None

    nlp = spacy.blank("en")
    try:
        nlp.add_pipe("abbreviation_detector")
    except Exception as exc:
        print(
            f"Warning: could not load abbreviation_detector ({exc}); continuing without abbreviation expansion.",
            file=sys.stderr,
        )
        return None
    return nlp


def get_passage_abbreviation_map(passage_text: str, abbr_nlp, debug: bool) -> dict[str, str]:
    if abbr_nlp is None:
        return {}

    passage_text = (passage_text or "").strip()
    if not passage_text:
        return {}

    try:
        doc = abbr_nlp(passage_text)
    except Exception as exc:
        debug_print(f"[DEBUG] Abbreviation parsing failed: {exc!r}", debug)
        return {}

    abbr_map: dict[str, str] = {}
    for abbr in getattr(doc._, "abbreviations", []) or []:
        short_form = str(abbr).strip()
        long_form_obj = getattr(abbr._, "long_form", None)
        long_form = str(long_form_obj).strip() if long_form_obj is not None else ""
        if short_form and long_form and short_form not in abbr_map:
            abbr_map[short_form] = long_form

    debug_print(f"[DEBUG] Abbreviation map: {abbr_map}", debug and bool(abbr_map))
    return abbr_map


def get_annotation_text(annotation: bioc.BioCAnnotation, passage: bioc.BioCPassage) -> str:
    text = (annotation.text or "").strip()
    if text:
        return text

    if not annotation.locations or passage.text is None:
        return ""

    location = annotation.locations[0]
    start = location.offset - passage.offset
    end = start + location.length
    if start < 0 or end > len(passage.text):
        return ""

    return passage.text[start:end].strip()


def normalize_with_pyobo(
    linker,
    mention_text: str,
    mention_nlp,
    topn: int,
    score_threshold: float,
    debug: bool,
    abbreviation_map: dict[str, str] | None = None,
) -> list[tuple[str, str, float]]:
    abbreviation_map = abbreviation_map or {}
    mention_text = (mention_text or "").strip()
    if not mention_text:
        return []

    candidate_texts: list[str] = []
    expanded = abbreviation_map.get(mention_text)
    if expanded and expanded != mention_text:
        candidate_texts.append(expanded)
    candidate_texts.append(mention_text)

    seen_ids: set[str] = set()
    all_results: list[tuple[str, str, float]] = []

    for text in candidate_texts:
        doc = mention_nlp(text)
        if not doc.ents and len(doc) > 0:
            span = doc.char_span(0, len(text), label="MENTION", alignment_mode="expand")
            if span is not None:
                doc.ents = [span]

        try:
            doc = linker(doc)
        except Exception as exc:
            print(f"Linking failed for mention {text!r}: {exc!r}", file=sys.stderr)
            continue

        if not doc.ents:
            continue

        ent = doc.ents[0]
        if not hasattr(ent._, "kb_ents") or not ent._.kb_ents:
            continue

        for identifier, score in ent._.kb_ents[:topn]:
            if score < score_threshold:
                continue

            normalized_identifier = normalize_identifier_case(identifier)
            canonical_name = normalized_identifier

            kb_entry = (
                linker.kb.cui_to_entity.get(identifier)
                or linker.kb.cui_to_entity.get(identifier.lower())
                or linker.kb.cui_to_entity.get(normalized_identifier)
                or linker.kb.cui_to_entity.get(normalized_identifier.lower())
            )
            if kb_entry is not None and getattr(kb_entry, "canonical_name", None):
                canonical_name = kb_entry.canonical_name

            if normalized_identifier in seen_ids:
                continue

            seen_ids.add(normalized_identifier)
            all_results.append((normalized_identifier, canonical_name, float(score)))

    all_results.sort(key=lambda item: item[2], reverse=True)
    debug_print(f"[DEBUG] {mention_text!r} -> {all_results[:topn]}", debug)
    return all_results[:topn]


def process_collection(input_xml: Path, output_xml: Path, linker, abbr_nlp, mention_nlp, args) -> None:
    with input_xml.open("r", encoding="utf-8") as handle:
        collection = bioc.load(handle)

    mention_cache: dict[tuple[str, tuple[tuple[str, str], ...]], list[tuple[str, str, float]]] = {}
    processed_annotations = 0
    matched_annotations = 0
    missing_annotations = 0
    debug_counter = 0

    for document in collection.documents:
        for passage in document.passages:
            if not passage.infons.get("annotatable", True):
                continue

            abbreviation_map = get_passage_abbreviation_map(passage.text or "", abbr_nlp, args.debug)
            abbr_items = tuple(sorted(abbreviation_map.items()))

            for annotation in passage.annotations:
                if annotation.infons.get("type") == "cell_vague":
                    continue

                processed_annotations += 1
                mention_text = get_annotation_text(annotation, passage)

                if args.debug and debug_counter < args.debug_max_print:
                    debug_print(
                        f"[DEBUG] Annotation #{processed_annotations}: "
                        f"type={annotation.infons.get('type')!r}, mention={mention_text!r}",
                        True,
                    )
                    debug_counter += 1

                if not mention_text:
                    missing_annotations += 1
                    continue

                cache_key = (mention_text, abbr_items)
                if cache_key not in mention_cache:
                    mention_cache[cache_key] = normalize_with_pyobo(
                        linker=linker,
                        mention_text=mention_text,
                        mention_nlp=mention_nlp,
                        topn=args.topn,
                        score_threshold=args.score_threshold,
                        debug=args.debug,
                        abbreviation_map=abbreviation_map,
                    )

                normalized_hits = mention_cache[cache_key]
                if not normalized_hits:
                    missing_annotations += 1
                    continue

                matched_annotations += 1
                for rank, (identifier, canonical_name, score) in enumerate(normalized_hits[: args.topn]):
                    annotation.infons[f"{MODEL_NAME}_id_{rank}"] = identifier
                    annotation.infons[f"{MODEL_NAME}_identifier_name_{rank}"] = canonical_name
                    annotation.infons[f"{MODEL_NAME}_identifier_score_{rank}"] = f"{score:.6f}"

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    with output_xml.open("w", encoding="utf-8") as handle:
        bioc.dump(collection, handle)

    print(f"Unique mention/context queries: {len(mention_cache)}")
    print(f"Processed annotations: {processed_annotations}")
    print(f"Matched annotations: {matched_annotations}")
    print(f"Missing annotations: {missing_annotations}")
    print(f"Saved output to: {output_xml}")


def main() -> int:
    args = parse_args()

    if not args.input_xml.is_file():
        print(f"Missing input XML: {args.input_xml}", file=sys.stderr)
        return 1
    if args.topn <= 0:
        print("--topn must be positive.", file=sys.stderr)
        return 1

    try:
        linker = load_celltype_linker(args.ontology_prefix, args.topn, args.debug)
        abbr_nlp = None if args.disable_abbreviations else build_abbreviation_nlp()
        mention_nlp = build_linking_nlp()
        process_collection(
            input_xml=args.input_xml,
            output_xml=args.output_xml,
            linker=linker,
            abbr_nlp=abbr_nlp,
            mention_nlp=mention_nlp,
            args=args,
        )
    except Exception as exc:
        print(f"SciSpaCy/PyOBO normalization failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
