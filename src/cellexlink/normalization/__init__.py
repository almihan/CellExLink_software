"""Cell Ontology normalization components for CellExLink."""

from __future__ import annotations

from .abbreviations import (
    AbbreviationCandidate,
    AbbreviationLookup,
    abbreviation_variant_keys,
    is_abbreviation_like,
    load_abbreviation_identifier_lookup,
    normalize_abbreviation_key,
)
from .linker import (
    DEFAULT_NEN_MODEL,
    CellOntologyLinker,
    MentionRecord,
    NormalizationCandidate,
    NormalizationResult,
    normalize_bioc,
    normalize_mentions,
)
from .ontology import (
    ConceptMetadata,
    TermEntry,
    default_ontology_path,
    load_cell_ontology_terms,
)
from .stemmer import normalize_text, plural_normalize_text
from .train import train_nen

__all__ = [
    "AbbreviationCandidate",
    "AbbreviationLookup",
    "CellOntologyLinker",
    "ConceptMetadata",
    "DEFAULT_NEN_MODEL",
    "MentionRecord",
    "NormalizationCandidate",
    "NormalizationResult",
    "TermEntry",
    "abbreviation_variant_keys",
    "default_ontology_path",
    "is_abbreviation_like",
    "load_abbreviation_identifier_lookup",
    "load_cell_ontology_terms",
    "normalize_abbreviation_key",
    "normalize_bioc",
    "normalize_mentions",
    "normalize_text",
    "plural_normalize_text",
    "train_nen",
]
