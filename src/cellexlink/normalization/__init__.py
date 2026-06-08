"""Cell Ontology normalization components for CellExLink."""

from __future__ import annotations

from .abbreviations import (
    ABBREVIATION_HEADER,
    AbbreviationCandidate,
    AbbreviationLookup,
    abbreviation_sequence_ratio,
    abbreviation_threshold_for,
    abbreviation_variant_keys,
    ab3p_fuzzy_threshold_for,
    build_document_abbreviation_lookup,
    default_abbreviations_path,
    is_abbreviation_like,
    load_abbreviation_identifier_lookup,
    normalize_abbreviation_key,
    normalize_pyab3p_output,
)
from .linker import (
    AMBIGUOUS_TOPN,
    DEFAULT_NEN_MODEL,
    DEFAULT_TOPN,
    CellOntologyLinker,
    EncoderHandle,
    MentionRecord,
    NormalizationCandidate,
    NormalizationResult,
    NormalizationRuntimeSummary,
    cosine_similarity_matrix,
    encode_texts,
    has_parenthetical_relation,
    is_sentence_transformers_model,
    load_encoder,
    normalize_bioc,
    normalize_mentions,
    resolve_model_label,
    sequence_ratio,
    token_jaccard,
)
from .ontology import (
    ConceptMetadata,
    TermEntry,
    default_ontology_path,
    load_cell_ontology_terms,
    load_terms,
)
from .stemmer import normalize_text, plural_normalize_text

try:
    from .train import train_nen
except Exception:  # pragma: no cover - training dependencies are optional.
    train_nen = None  # type: ignore

__all__ = [
    "ABBREVIATION_HEADER",
    "AMBIGUOUS_TOPN",
    "DEFAULT_NEN_MODEL",
    "DEFAULT_TOPN",
    "AbbreviationCandidate",
    "AbbreviationLookup",
    "CellOntologyLinker",
    "ConceptMetadata",
    "EncoderHandle",
    "MentionRecord",
    "NormalizationCandidate",
    "NormalizationResult",
    "NormalizationRuntimeSummary",
    "TermEntry",
    "ab3p_fuzzy_threshold_for",
    "abbreviation_sequence_ratio",
    "abbreviation_threshold_for",
    "abbreviation_variant_keys",
    "build_document_abbreviation_lookup",
    "cosine_similarity_matrix",
    "default_abbreviations_path",
    "default_ontology_path",
    "encode_texts",
    "has_parenthetical_relation",
    "is_abbreviation_like",
    "is_sentence_transformers_model",
    "load_abbreviation_identifier_lookup",
    "load_cell_ontology_terms",
    "load_encoder",
    "load_terms",
    "normalize_abbreviation_key",
    "normalize_bioc",
    "normalize_mentions",
    "normalize_pyab3p_output",
    "normalize_text",
    "plural_normalize_text",
    "resolve_model_label",
    "sequence_ratio",
    "token_jaccard",
    "train_nen",
]
