"""scispaCy baseline runner for CellExLink CellLink benchmarks."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import bioc
from bioc import biocxml


ROOT = Path(__file__).resolve().parents[3]

DEFAULT_OUTPUT_DIR = ROOT / "benchmarks" / "benchmark_outputs" / "scispacy"
DEFAULT_DATASET_NAME = "Celllink"
DEFAULT_SCISPACY_MODEL = "en_ner_craft_md"
SCISPACY_LINKER_NAME = "scispacy"
TARGET_SCISPACY_LABEL = "CL"
OUTPUT_ENTITY_TYPE = "cell_type"

EVAL_NER = ROOT / "benchmarks" / "evaluate_ner.py"
EVAL_NEN = ROOT / "benchmarks" / "evaluate_nen.py"
EVAL_E2E = ROOT / "benchmarks" / "evaluate_end_to_end.py"


def default_celllink_xml() -> Path:
    """Return the default CellLink XML path, supporting both common spellings."""
    candidates = [
        ROOT / "benchmarks" / "data" / "evaluation" / "Celllink" / "test.xml",
        ROOT / "benchmarks" / "data" / "evaluation" / "CellLink" / "test.xml",
    ]

    for path in candidates:
        if path.is_file():
            return path

    return candidates[0]


@dataclass(slots=True)
class TextUnit:
    container: object
    text: str
    base_offset: int
    context: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the minimal scispaCy CellLink benchmark baseline."
    )

    parser.add_argument(
        "--mode",
        choices=("all", "ner", "nen", "e2e"),
        default="all",
        help="Benchmark mode to run. Default: all.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_celllink_xml(),
        help="CellLink BioC XML input. Default: benchmarks/data/evaluation/Celllink/test.xml",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=None,
        help="Gold BioC XML for evaluation. Default: same as --input.",
    )
    parser.add_argument(
        "--nen-input",
        type=Path,
        default=None,
        help="Gold-span XML for NEN-only mode. Default: same as --input.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory. Default: benchmarks/benchmark_outputs/scispacy/",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Dataset name used in evaluator arguments. Default: Celllink.",
    )

    parser.add_argument("--model-name", default=DEFAULT_SCISPACY_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--offset-mode", choices=("char", "bioc_bytes"), default="char")

    parser.add_argument("--ontology-prefix", default="cl")
    parser.add_argument("--topn", type=int, default=10)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--topk", choices=("1", "5", "10", "all"), default="1")
    parser.add_argument("--disable-abbreviations", action="store_true")
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def require_int_offset(value, context: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid offset for {context}: {value!r}") from exc


def build_char_to_byte_map(text: str) -> list[int]:
    offsets = [0]
    total = 0

    for char in text:
        total += len(char.encode("utf-8"))
        offsets.append(total)

    return offsets


def text_length_in_offset_units(text: str, offset_mode: str) -> int:
    if offset_mode == "char":
        return len(text)

    return build_char_to_byte_map(text)[-1]


def local_offset_to_char_index(
    text: str,
    local_offset: int,
    offset_mode: str,
    context: str,
    char_to_byte: list[int] | None = None,
) -> int:
    if local_offset < 0:
        raise ValueError(f"Negative local offset in {context}: {local_offset}")

    if offset_mode == "char":
        if local_offset > len(text):
            raise ValueError(
                f"Local character offset out of range in {context}: "
                f"{local_offset} > {len(text)}"
            )

        return local_offset

    if char_to_byte is None:
        char_to_byte = build_char_to_byte_map(text)

    position = bisect_left(char_to_byte, local_offset)

    if position == len(char_to_byte) or char_to_byte[position] != local_offset:
        raise ValueError(
            f"Local byte offset {local_offset} does not land on a UTF-8 "
            f"character boundary in {context}."
        )

    return position


def extract_sentence_text(
    passage,
    sentences: list,
    sentence_index: int,
    offset_mode: str,
    doc_id: str,
    passage_index: int,
) -> str:
    sentence = sentences[sentence_index]
    sentence_text = getattr(sentence, "text", None)

    if sentence_text is not None:
        return sentence_text

    passage_text = getattr(passage, "text", None) or ""

    if not passage_text:
        return ""

    context = f"document={doc_id} passage={passage_index} sentence={sentence_index}"
    passage_offset = require_int_offset(
        getattr(passage, "offset", None),
        f"{context} passage offset",
    )
    sentence_offset = require_int_offset(
        getattr(sentence, "offset", None),
        f"{context} sentence offset",
    )

    if sentence_index + 1 < len(sentences):
        next_offset = require_int_offset(
            getattr(sentences[sentence_index + 1], "offset", None),
            f"{context} next sentence offset",
        )
    else:
        next_offset = passage_offset + text_length_in_offset_units(
            passage_text,
            offset_mode,
        )

    local_start = sentence_offset - passage_offset
    local_end = next_offset - passage_offset

    if local_end < local_start:
        raise ValueError(
            f"Invalid sentence offsets in {context}: "
            f"start={sentence_offset}, end={next_offset}"
        )

    char_to_byte = build_char_to_byte_map(passage_text) if offset_mode == "bioc_bytes" else None

    start_char = local_offset_to_char_index(
        passage_text,
        local_start,
        offset_mode,
        context,
        char_to_byte,
    )
    end_char = local_offset_to_char_index(
        passage_text,
        local_end,
        offset_mode,
        context,
        char_to_byte,
    )

    return passage_text[start_char:end_char]


def iter_text_units(collection, offset_mode: str) -> Iterator[TextUnit]:
    for doc_index, document in enumerate(collection.documents):
        doc_id = str(getattr(document, "id", None) or doc_index)

        for passage_index, passage in enumerate(document.passages):
            sentences = list(getattr(passage, "sentences", []) or [])

            if sentences:
                for sentence_index, sentence in enumerate(sentences):
                    text = extract_sentence_text(
                        passage=passage,
                        sentences=sentences,
                        sentence_index=sentence_index,
                        offset_mode=offset_mode,
                        doc_id=doc_id,
                        passage_index=passage_index,
                    )

                    if not text:
                        continue

                    yield TextUnit(
                        container=sentence,
                        text=text,
                        base_offset=require_int_offset(
                            getattr(sentence, "offset", None),
                            f"document={doc_id} passage={passage_index} "
                            f"sentence={sentence_index}",
                        ),
                        context=(
                            f"document={doc_id} passage={passage_index} "
                            f"sentence={sentence_index}"
                        ),
                    )
            else:
                passage_text = getattr(passage, "text", None) or ""

                if not passage_text:
                    continue

                yield TextUnit(
                    container=passage,
                    text=passage_text,
                    base_offset=require_int_offset(
                        getattr(passage, "offset", None),
                        f"document={doc_id} passage={passage_index}",
                    ),
                    context=f"document={doc_id} passage={passage_index}",
                )


def clear_existing_annotations_and_relations(collection) -> None:
    for document in collection.documents:
        if hasattr(document, "relations"):
            document.relations = []

        for passage in document.passages:
            if hasattr(passage, "annotations"):
                passage.annotations = []
            if hasattr(passage, "relations"):
                passage.relations = []

            for sentence in list(getattr(passage, "sentences", []) or []):
                if hasattr(sentence, "annotations"):
                    sentence.annotations = []
                if hasattr(sentence, "relations"):
                    sentence.relations = []


def add_location(annotation, location) -> None:
    add_location_fn = getattr(annotation, "add_location", None)

    if callable(add_location_fn):
        add_location_fn(location)
    else:
        annotation.locations.append(location)


def add_annotation(container, annotation) -> None:
    add_annotation_fn = getattr(container, "add_annotation", None)

    if callable(add_annotation_fn):
        add_annotation_fn(annotation)
    else:
        container.annotations.append(annotation)


def load_scispacy_ner_model(model_name: str):
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError(
            "spaCy/scispaCy is not installed. Activate the scispaCy baseline "
            "environment first."
        ) from exc

    try:
        nlp = spacy.load(model_name)
    except OSError as exc:
        raise RuntimeError(f"Could not load scispaCy model {model_name!r}.") from exc

    if "ner" not in nlp.pipe_names:
        raise RuntimeError(f"Model {model_name!r} does not contain an NER component.")

    ner_labels = set(getattr(nlp.get_pipe("ner"), "labels", ()))

    if ner_labels and TARGET_SCISPACY_LABEL not in ner_labels:
        print(
            f"WARNING: model {model_name!r} does not list label "
            f"{TARGET_SCISPACY_LABEL!r}. Output may be empty.",
            file=sys.stderr,
        )

    return nlp


def run_ner_prediction(
    input_xml: Path,
    output_xml: Path,
    *,
    model_name: str,
    batch_size: int,
    offset_mode: str,
) -> None:
    print(f"Running scispaCy NER: {input_xml} -> {output_xml}")

    nlp = load_scispacy_ner_model(model_name)

    with input_xml.open("r", encoding="utf-8") as handle:
        collection = biocxml.load(handle)

    clear_existing_annotations_and_relations(collection)
    units = list(iter_text_units(collection, offset_mode=offset_mode))

    annotation_index = 1
    num_entities = 0

    for unit, doc in zip(units, nlp.pipe((u.text for u in units), batch_size=batch_size)):
        char_to_byte = build_char_to_byte_map(unit.text) if offset_mode == "bioc_bytes" else None

        for ent in doc.ents:
            if ent.label_ != TARGET_SCISPACY_LABEL:
                continue

            start_char = int(ent.start_char)
            end_char = int(ent.end_char)

            if start_char >= end_char:
                continue

            if offset_mode == "bioc_bytes":
                assert char_to_byte is not None
                start = unit.base_offset + char_to_byte[start_char]
                length = char_to_byte[end_char] - char_to_byte[start_char]
            else:
                start = unit.base_offset + start_char
                length = end_char - start_char

            if length <= 0:
                continue

            annotation = bioc.BioCAnnotation()
            annotation.id = f"T{annotation_index}"
            annotation.infons["type"] = OUTPUT_ENTITY_TYPE
            annotation.text = unit.text[start_char:end_char]

            add_location(annotation, bioc.BioCLocation(start, length))
            add_annotation(unit.container, annotation)

            annotation_index += 1
            num_entities += 1

    output_xml.parent.mkdir(parents=True, exist_ok=True)

    with output_xml.open("w", encoding="utf-8") as handle:
        biocxml.dump(collection, handle)

    print(f"Processed text units: {len(units)}")
    print(f"Predicted CL entities: {num_entities}")
    print(f"Wrote NER XML: {output_xml}")


def normalize_identifier_case(identifier: str) -> str:
    identifier = (identifier or "").strip()
    return re.sub(r"^cl[_:]", "CL:", identifier, flags=re.IGNORECASE)


def load_celltype_linker(ontology_prefix: str, topn: int, debug: bool):
    try:
        import pyobo
        from scispacy.linking import EntityLinker
    except ImportError as exc:
        raise RuntimeError(
            "pyobo and scispaCy are required for this optional baseline."
        ) from exc

    if debug:
        print(f"[DEBUG] Loading PyOBO ontology prefix: {ontology_prefix}")

    try:
        return pyobo.get_scispacy_entity_linker(
            ontology_prefix,
            filter_for_definitions=False,
            max_entities_per_mention=topn,
        )
    except Exception as first_exc:
        print(
            f"High-level PyOBO linker construction failed for "
            f"{ontology_prefix!r}: {first_exc!r}",
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
            f"Could not load PyOBO/scispaCy linker for ontology "
            f"{ontology_prefix!r}."
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
            "Warning: scispacy.abbreviation is unavailable; continuing without "
            "abbreviation expansion.",
            file=sys.stderr,
        )
        return None

    nlp = spacy.blank("en")

    try:
        nlp.add_pipe("abbreviation_detector")
    except Exception as exc:
        print(
            f"Warning: could not load abbreviation_detector ({exc}); "
            "continuing without abbreviation expansion.",
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
        if debug:
            print(f"[DEBUG] Abbreviation parsing failed: {exc!r}")
        return {}

    abbr_map: dict[str, str] = {}

    for abbr in getattr(doc._, "abbreviations", []) or []:
        short_form = str(abbr).strip()
        long_form_obj = getattr(abbr._, "long_form", None)
        long_form = str(long_form_obj).strip() if long_form_obj is not None else ""

        if short_form and long_form and short_form not in abbr_map:
            abbr_map[short_form] = long_form

    if debug and abbr_map:
        print(f"[DEBUG] Abbreviation map: {abbr_map}")

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
    *,
    topn: int,
    score_threshold: float,
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

    return all_results[:topn]


def run_nen_prediction(
    input_xml: Path,
    output_xml: Path,
    *,
    ontology_prefix: str,
    topn: int,
    score_threshold: float,
    disable_abbreviations: bool,
    debug: bool,
) -> None:
    print(f"Running scispaCy NEN: {input_xml} -> {output_xml}")

    linker = load_celltype_linker(ontology_prefix, topn, debug)
    mention_nlp = build_linking_nlp()
    abbr_nlp = None if disable_abbreviations else build_abbreviation_nlp()

    with input_xml.open("r", encoding="utf-8") as handle:
        collection = bioc.load(handle)

    mention_cache: dict[tuple[str, tuple[tuple[str, str], ...]], list[tuple[str, str, float]]] = {}
    processed_annotations = 0
    matched_annotations = 0
    missing_annotations = 0

    for document in collection.documents:
        for passage in document.passages:
            abbreviation_map = get_passage_abbreviation_map(
                passage.text or "",
                abbr_nlp,
                debug,
            )
            abbr_items = tuple(sorted(abbreviation_map.items()))

            for annotation in passage.annotations:
                if annotation.infons.get("type") == "cell_vague":
                    continue

                processed_annotations += 1
                mention_text = get_annotation_text(annotation, passage)

                if not mention_text:
                    missing_annotations += 1
                    continue

                cache_key = (mention_text, abbr_items)

                if cache_key not in mention_cache:
                    mention_cache[cache_key] = normalize_with_pyobo(
                        linker,
                        mention_text,
                        mention_nlp,
                        topn=topn,
                        score_threshold=score_threshold,
                        abbreviation_map=abbreviation_map,
                    )

                normalized_hits = mention_cache[cache_key]

                if not normalized_hits:
                    missing_annotations += 1
                    continue

                matched_annotations += 1

                for rank, (identifier, canonical_name, score) in enumerate(normalized_hits[:topn]):
                    annotation.infons[f"{SCISPACY_LINKER_NAME}_id_{rank}"] = identifier
                    annotation.infons[
                        f"{SCISPACY_LINKER_NAME}_identifier_name_{rank}"
                    ] = canonical_name
                    annotation.infons[
                        f"{SCISPACY_LINKER_NAME}_identifier_score_{rank}"
                    ] = f"{score:.6f}"

    output_xml.parent.mkdir(parents=True, exist_ok=True)

    with output_xml.open("w", encoding="utf-8") as handle:
        bioc.dump(collection, handle)

    print(f"Unique mention/context queries: {len(mention_cache)}")
    print(f"Processed annotations: {processed_annotations}")
    print(f"Matched annotations: {matched_annotations}")
    print(f"Missing annotations: {missing_annotations}")
    print(f"Wrote normalized XML: {output_xml}")


def output_prefix(dataset_name: str, input_xml: Path) -> str:
    return f"{dataset_name}_{input_xml.stem}"


def evaluate_ner(gold_xml: Path, pred_xml: Path, output_csv: Path, dataset_name: str) -> None:
    run_command(
        [
            sys.executable,
            str(EVAL_NER),
            "--system",
            "ScispaCy",
            "--gold",
            f"{dataset_name}={gold_xml}",
            "--pred",
            f"{dataset_name}={pred_xml}",
            "--output-csv",
            str(output_csv),
        ]
    )


def evaluate_nen(gold_xml: Path, pred_xml: Path, output_csv: Path, dataset_name: str, args) -> None:
    run_command(
        [
            sys.executable,
            str(EVAL_NEN),
            "--gold",
            f"{dataset_name}={gold_xml}",
            "--pred",
            f"{dataset_name}={pred_xml}",
            "--dataset-style",
            "celllink",
            "--model-names",
            SCISPACY_LINKER_NAME,
            "--topk",
            args.topk,
            "--threshold",
            str(args.score_threshold),
            "--output-csv",
            str(output_csv),
        ]
    )


def evaluate_e2e(gold_xml: Path, pred_xml: Path, output_csv: Path, dataset_name: str, args) -> None:
    run_command(
        [
            sys.executable,
            str(EVAL_E2E),
            "--gold",
            f"{dataset_name}={gold_xml}",
            "--pred",
            f"{dataset_name}={pred_xml}",
            "--dataset-style",
            "celllink",
            "--model-names",
            SCISPACY_LINKER_NAME,
            "--topk",
            args.topk,
            "--threshold",
            str(args.score_threshold),
            "--output-csv",
            str(output_csv),
        ]
    )


def run_ner_mode(args, input_xml: Path, gold_xml: Path) -> None:
    prefix = output_prefix(args.dataset_name, input_xml)
    ner_xml = args.output_dir / "ner" / f"{prefix}.scispacy.ner.xml"
    result_csv = args.output_dir / "scispacy_celllink_ner.csv"

    run_ner_prediction(
        input_xml,
        ner_xml,
        model_name=args.model_name,
        batch_size=args.batch_size,
        offset_mode=args.offset_mode,
    )
    evaluate_ner(gold_xml, ner_xml, result_csv, args.dataset_name)


def run_nen_mode(args, nen_input_xml: Path, gold_xml: Path) -> None:
    prefix = output_prefix(args.dataset_name, nen_input_xml)
    normalized_xml = args.output_dir / "nen" / f"{prefix}.scispacy.normalized.xml"
    result_csv = args.output_dir / "scispacy_celllink_nen.csv"

    run_nen_prediction(
        nen_input_xml,
        normalized_xml,
        ontology_prefix=args.ontology_prefix,
        topn=args.topn,
        score_threshold=args.score_threshold,
        disable_abbreviations=args.disable_abbreviations,
        debug=args.debug,
    )
    evaluate_nen(gold_xml, normalized_xml, result_csv, args.dataset_name, args)


def run_e2e_mode(args, input_xml: Path, gold_xml: Path) -> None:
    prefix = output_prefix(args.dataset_name, input_xml)
    ner_xml = args.output_dir / "end_to_end" / f"{prefix}.scispacy.ner.xml"
    normalized_xml = args.output_dir / "end_to_end" / f"{prefix}.scispacy.normalized.xml"
    result_csv = args.output_dir / "scispacy_celllink_end_to_end.csv"

    run_ner_prediction(
        input_xml,
        ner_xml,
        model_name=args.model_name,
        batch_size=args.batch_size,
        offset_mode=args.offset_mode,
    )
    run_nen_prediction(
        ner_xml,
        normalized_xml,
        ontology_prefix=args.ontology_prefix,
        topn=args.topn,
        score_threshold=args.score_threshold,
        disable_abbreviations=args.disable_abbreviations,
        debug=args.debug,
    )
    evaluate_e2e(gold_xml, normalized_xml, result_csv, args.dataset_name, args)


def main() -> int:
    args = parse_args()

    input_xml = args.input.resolve()
    gold_xml = args.gold.resolve() if args.gold else input_xml
    nen_input_xml = args.nen_input.resolve() if args.nen_input else input_xml
    args.output_dir = args.output_dir.resolve()

    if not input_xml.is_file():
        print(f"Missing input XML: {input_xml}", file=sys.stderr)
        return 1

    if not gold_xml.is_file():
        print(f"Missing gold XML: {gold_xml}", file=sys.stderr)
        return 1

    if not nen_input_xml.is_file():
        print(f"Missing NEN input XML: {nen_input_xml}", file=sys.stderr)
        return 1

    if args.batch_size <= 0:
        print("--batch-size must be positive.", file=sys.stderr)
        return 1

    if args.topn <= 0:
        print("--topn must be positive.", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Mode: {args.mode}")
    print(f"Input XML: {input_xml}")
    print(f"NEN input XML: {nen_input_xml}")
    print(f"Gold XML: {gold_xml}")
    print(f"Output directory: {args.output_dir}")

    try:
        if args.mode in {"all", "ner"}:
            run_ner_mode(args, input_xml, gold_xml)

        if args.mode in {"all", "nen"}:
            run_nen_mode(args, nen_input_xml, gold_xml)

        if args.mode in {"all", "e2e"}:
            run_e2e_mode(args, input_xml, gold_xml)

    except subprocess.CalledProcessError as exc:
        print(f"Evaluator failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode
    except Exception as exc:
        print(f"scispaCy baseline failed: {exc}", file=sys.stderr)
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
