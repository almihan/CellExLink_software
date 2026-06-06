"""
Cell Ontology resource loading for CellExLink normalization.

Expected JSONL resource format, matching the current CellExLink resource:

{
  "norm_concept_id": "CL:0000077",
  "norm_preferred_label": "mesothelial cell",
  "synonyms": ["mesotheliocyte", ...],
  "namespace": "cell"
}

The loader is permissive and also accepts common aliases such as id, label,
preferred_label, aliases and names. This helps when rebuilding the ontology
resource from BioPortal, OBO, OWL, PyOBO, or a curated table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, Optional

from .stemmer import normalize_text

DEFAULT_ONTOLOGY_FILENAME = "cell_ontology_v2025-12-17.jsonl"


@dataclass(frozen=True, slots=True)
class TermEntry:
    """One searchable alias/name for one Cell Ontology concept."""

    name: str
    raw_name: str
    identifier: str
    preferred_label: str
    is_preferred: bool = False
    namespace: str = ""
    source: str = "ontology"


@dataclass(slots=True)
class ConceptMetadata:
    """Metadata and names for one ontology concept."""

    identifier: str
    preferred_label: str
    synonyms: set[str] = field(default_factory=set)
    names: set[str] = field(default_factory=set)
    namespace: str = ""

    def add_name(self, name: str, *, synonym: bool = True) -> None:
        cleaned = str(name).strip()
        if not cleaned:
            return
        self.names.add(cleaned)
        if synonym and cleaned != self.preferred_label:
            self.synonyms.add(cleaned)


class OntologyFormatError(ValueError):
    """Raised when a Cell Ontology JSONL resource cannot be parsed."""


def default_ontology_path() -> Path:
    """
    Return the installed package's default Cell Ontology JSONL path.

    This assumes you copied the JSONL resource into:

        src/cellexlink/resources/cell_ontology_v2025-12-17.jsonl
    """
    candidate = resources.files("cellexlink").joinpath(
        "resources", DEFAULT_ONTOLOGY_FILENAME
    )
    return Path(str(candidate))


def _first_nonempty(record: dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def parse_ontology_record(record: dict[str, Any]) -> tuple[str, str, list[str], str]:
    """Extract identifier, preferred label, synonyms and namespace from one record."""
    identifier = _first_nonempty(
        record,
        [
            "norm_concept_id",
            "concept_id",
            "identifier",
            "id",
            "cl_id",
            "obo_id",
        ],
    )
    preferred_label = _first_nonempty(
        record,
        [
            "norm_preferred_label",
            "preferred_label",
            "label",
            "name",
            "term",
        ],
    )

    if not identifier or not preferred_label:
        raise OntologyFormatError(
            "Ontology records must contain an identifier and preferred label. "
            f"Bad record: {record!r}"
        )

    synonyms: list[str] = []
    for key in ["synonyms", "aliases", "exact_synonyms", "related_synonyms", "names"]:
        synonyms.extend(_as_list(record.get(key)))

    # Some resources store one alias per row.
    alias = _first_nonempty(record, ["alias", "synonym"])
    if alias:
        synonyms.append(alias)

    namespace = str(record.get("namespace", "") or "").strip()
    return identifier, preferred_label, synonyms, namespace


def load_cell_ontology_terms(
    path: str | Path,
    *,
    include_preferred_labels: bool = True,
    include_synonyms: bool = True,
    deduplicate: bool = True,
) -> tuple[list[TermEntry], dict[str, ConceptMetadata]]:
    """
    Load a Cell Ontology alias dictionary from JSONL.

    Returns
    -------
    term_entries:
        Searchable ontology names/synonyms for dense retrieval.
    concept_metadata:
        Per-concept metadata used for re-ranking.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Cell Ontology JSONL file does not exist: {path}")

    term_entries: list[TermEntry] = []
    concept_metadata: dict[str, ConceptMetadata] = {}
    seen_entries: set[tuple[str, str, bool]] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise OntologyFormatError(
                    f"Invalid JSON on line {line_no} of {path}: {exc}"
                ) from exc

            identifier, preferred_label, synonyms, namespace = parse_ontology_record(record)

            if identifier not in concept_metadata:
                concept_metadata[identifier] = ConceptMetadata(
                    identifier=identifier,
                    preferred_label=preferred_label,
                    namespace=namespace,
                )

            meta = concept_metadata[identifier]
            meta.add_name(preferred_label, synonym=False)

            if include_preferred_labels:
                entry = TermEntry(
                    name=normalize_text(preferred_label),
                    raw_name=preferred_label,
                    identifier=identifier,
                    preferred_label=preferred_label,
                    is_preferred=True,
                    namespace=namespace,
                )
                key = (entry.identifier, entry.name, entry.is_preferred)
                if not deduplicate or key not in seen_entries:
                    term_entries.append(entry)
                    seen_entries.add(key)

            if include_synonyms:
                for synonym in synonyms:
                    synonym = str(synonym).strip()
                    if not synonym:
                        continue
                    meta.add_name(synonym, synonym=True)
                    entry = TermEntry(
                        name=normalize_text(synonym),
                        raw_name=synonym,
                        identifier=identifier,
                        preferred_label=preferred_label,
                        is_preferred=False,
                        namespace=namespace,
                    )
                    key = (entry.identifier, entry.name, entry.is_preferred)
                    if not deduplicate or key not in seen_entries:
                        term_entries.append(entry)
                        seen_entries.add(key)

    if not term_entries:
        raise OntologyFormatError(f"No usable ontology terms were loaded from {path}")

    return term_entries, concept_metadata


def write_cell_ontology_terms(
    records: Iterable[dict[str, Any]],
    output_jsonl: str | Path,
) -> Path:
    """
    Write ontology records to the JSONL schema used by CellExLink.

    This is useful for scripts/build_cell_ontology_resource.py later.
    """
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            identifier, preferred_label, synonyms, namespace = parse_ontology_record(record)
            out = {
                "norm_concept_id": identifier,
                "norm_preferred_label": preferred_label,
                "synonyms": sorted(set(synonyms)),
                "namespace": namespace,
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")

    return output_jsonl
