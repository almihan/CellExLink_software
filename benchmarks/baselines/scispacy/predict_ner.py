#!/usr/bin/env python3
"""Run ScispaCy NER on BioC XML and write cell-type span predictions.

This script is an optional baseline script for CellExLink benchmarks. It is not
part of the installable CellExLink package.

The default ScispaCy model is `en_ner_craft_md`, which has a `CL` entity label.
Only entities with that label are written to the output BioC file as
`type=cell_type`.
"""

from __future__ import annotations

import argparse
import os
import sys
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import bioc
from bioc import biocxml


DEFAULT_MODEL_NAME = "en_ner_craft_md"
TARGET_SCISPACY_LABEL = "CL"
OUTPUT_ENTITY_TYPE = "cell_type"


@dataclass(slots=True)
class TextUnit:
    """A passage or sentence that should be processed by ScispaCy."""

    container: object
    text: str
    base_offset: int
    context: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ScispaCy NER on BioC XML and write BioC XML predictions."
    )
    parser.add_argument("--input-xml", type=Path, required=True, help="Input BioC XML file.")
    parser.add_argument("--output-xml", type=Path, required=True, help="Output BioC XML file.")
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=f"ScispaCy model to load. Default: {DEFAULT_MODEL_NAME}",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for nlp.pipe().")
    parser.add_argument(
        "--offset-mode",
        choices=("char", "bioc_bytes"),
        default="char",
        help=(
            "Interpret BioC offsets as Python character offsets or UTF-8 byte offsets. "
            "Most XML files created by this repository use char offsets."
        ),
    )
    return parser.parse_args()


def load_scispacy_model(model_name: str):
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError(
            "spaCy/scispaCy is not installed. Create the optional ScispaCy baseline "
            "environment first."
        ) from exc

    try:
        nlp = spacy.load(model_name)
    except OSError as exc:
        raise RuntimeError(
            f"Could not load ScispaCy model {model_name!r}. Install the model first."
        ) from exc

    if "ner" not in nlp.pipe_names:
        raise RuntimeError(f"Model {model_name!r} does not contain an NER component.")

    ner_labels = set(getattr(nlp.get_pipe("ner"), "labels", ()))
    if ner_labels and TARGET_SCISPACY_LABEL not in ner_labels:
        print(
            f"WARNING: model {model_name!r} does not list label {TARGET_SCISPACY_LABEL!r}. "
            "Output may be empty.",
            file=sys.stderr,
        )

    return nlp


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
            f"Local byte offset {local_offset} does not land on a UTF-8 character "
            f"boundary in {context}."
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
        next_offset = passage_offset + text_length_in_offset_units(passage_text, offset_mode)

    local_start = sentence_offset - passage_offset
    local_end = next_offset - passage_offset
    if local_end < local_start:
        raise ValueError(
            f"Invalid sentence offsets in {context}: start={sentence_offset}, end={next_offset}"
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
                            f"document={doc_id} passage={passage_index} sentence={sentence_index}",
                        ),
                        context=f"document={doc_id} passage={passage_index} sentence={sentence_index}",
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


def predict_cl_entities_into_collection(collection, nlp, batch_size: int, offset_mode: str) -> tuple[int, int]:
    clear_existing_annotations_and_relations(collection)
    units = list(iter_text_units(collection, offset_mode=offset_mode))
    if not units:
        return 0, 0

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

    return len(units), num_entities


def main() -> int:
    args = parse_args()
    args.input_xml = args.input_xml.resolve()
    args.output_xml = args.output_xml.resolve()

    if not args.input_xml.is_file():
        print(f"Missing input XML: {args.input_xml}", file=sys.stderr)
        return 1
    if args.batch_size <= 0:
        print("--batch-size must be a positive integer.", file=sys.stderr)
        return 1

    args.output_xml.parent.mkdir(parents=True, exist_ok=True)

    print(f"SciSpaCy model: {args.model_name}")
    print(f"Input XML: {args.input_xml}")
    print(f"Output XML: {args.output_xml}")
    print(f"Offset mode: {args.offset_mode}")
    print(f"Batch size: {args.batch_size}")
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    try:
        nlp = load_scispacy_model(args.model_name)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    try:
        with args.input_xml.open("r", encoding="utf-8") as handle:
            collection = biocxml.load(handle)
    except Exception as exc:
        print(f"Failed to read BioC XML: {exc}", file=sys.stderr)
        return 1

    try:
        num_units, num_entities = predict_cl_entities_into_collection(
            collection=collection,
            nlp=nlp,
            batch_size=args.batch_size,
            offset_mode=args.offset_mode,
        )
    except Exception as exc:
        print(f"SciSpaCy prediction failed: {exc}", file=sys.stderr)
        return 1

    try:
        with args.output_xml.open("w", encoding="utf-8") as handle:
            biocxml.dump(collection, handle)
    except Exception as exc:
        print(f"Failed to write BioC XML: {exc}", file=sys.stderr)
        return 1

    print(f"Processed text units: {num_units}")
    print(f"Predicted CL entities: {num_entities}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
