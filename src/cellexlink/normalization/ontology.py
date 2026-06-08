"""Cell Ontology alias loading for CellExLink NEN.

The default loader is deliberately strict.  It expects the same JSONL schema
used by the original CellExLink normalizer:

    norm_concept_id, norm_preferred_label, synonyms, namespace

Keeping the schema and alias order stable is important for reproducing NEN
results from the original code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Iterable, Optional

from .stemmer import plural_normalize_text

DEFAULT_ONTOLOGY_FILENAME = "cell_ontology_v2025-12-17.jsonl"


@dataclass(slots=True)
class TermEntry:
    """One searchable ontology alias."""

    name: str
    raw_name: str
    identifier: str
    preferred_label: str
    is_preferred: bool = False


@dataclass(slots=True)
class ConceptMetadata:
    """Metadata grouped by Cell Ontology concept identifier."""

    preferred_label: str
    synonyms: set[str] = field(default_factory=set)
    names: set[str] = field(default_factory=set)
    namespace: str = ""


def default_ontology_path() -> Path:
    """Return the packaged Cell Ontology JSONL resource path."""

    candidate = resources.files("cellexlink").joinpath(
        "resources", DEFAULT_ONTOLOGY_FILENAME
    )
    return Path(str(candidate))


def _as_namespace_set(namespace_filter: Optional[str | Iterable[str]]) -> Optional[set[str]]:
    if namespace_filter is None:
        return None
    if isinstance(namespace_filter, str):
        return {namespace_filter}
    return {str(item) for item in namespace_filter}


def load_cell_ontology_terms(
    ontology_path: str | Path,
    *,
    namespace_filter: Optional[str | Iterable[str]] = None,
) -> tuple[list[TermEntry], dict[str, ConceptMetadata]]:
    """Load ontology aliases using the original CellExLink behavior.

    Notes
    -----
    * Preferred labels are added before synonyms.
    * Alias strings are plural-normalized before embedding.
    * Term entries are not deduplicated, so retrieval order remains compatible
      with the original JSONL order.
    * Alternative JSON field names are intentionally not accepted here; this is
      a benchmark/reproducibility path, not a general ontology converter.
    """

    path = Path(ontology_path)
    if not path.is_file():
        raise FileNotFoundError(f"Cell Ontology JSONL file does not exist: {path}")

    namespaces = _as_namespace_set(namespace_filter)
    term_entries: list[TermEntry] = []
    concept_metadata: dict[str, ConceptMetadata] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON on line {line_no} in {path}: {exc}") from exc

            identifier = record.get("norm_concept_id")
            preferred_label = record.get("norm_preferred_label")
            synonyms = record.get("synonyms", []) or []
            namespace = record.get("namespace", "") or ""

            if not identifier or not preferred_label:
                continue
            if namespaces is not None and namespace not in namespaces:
                continue
            if not isinstance(synonyms, list):
                raise ValueError(
                    f"Expected 'synonyms' to be a list on line {line_no} in {path}"
                )

            if identifier not in concept_metadata:
                concept_metadata[identifier] = ConceptMetadata(
                    preferred_label=str(preferred_label),
                    namespace=str(namespace),
                )

            meta = concept_metadata[identifier]
            meta.names.add(str(preferred_label))

            term_entries.append(
                TermEntry(
                    name=plural_normalize_text(preferred_label),
                    raw_name=str(preferred_label),
                    identifier=str(identifier),
                    preferred_label=str(preferred_label),
                    is_preferred=True,
                )
            )

            for synonym in synonyms:
                if not synonym:
                    continue
                synonym = str(synonym)
                meta.synonyms.add(synonym)
                meta.names.add(synonym)
                term_entries.append(
                    TermEntry(
                        name=plural_normalize_text(synonym),
                        raw_name=synonym,
                        identifier=str(identifier),
                        preferred_label=str(preferred_label),
                        is_preferred=False,
                    )
                )

    return term_entries, concept_metadata


# Backward-compatible short name used by some scripts.
load_terms = load_cell_ontology_terms


__all__ = [
    "DEFAULT_ONTOLOGY_FILENAME",
    "TermEntry",
    "ConceptMetadata",
    "default_ontology_path",
    "load_cell_ontology_terms",
    "load_terms",
]
