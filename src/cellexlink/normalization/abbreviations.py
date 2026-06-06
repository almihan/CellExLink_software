"""
Abbreviation handling for CellExLink normalization.

CellExLink uses two abbreviation signals:
1. A corpus-derived short-form -> CL identifier TSV.
2. Optional document-level long-form recovery via pyab3p for ambiguous short forms.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Optional

DASH_PATTERN = r"[\-\u2010\u2011\u2012\u2013\u2014\u2212]"
ABBREVIATION_HEADER = ["short_form", "matched_cl_id"]

try:  # Optional dependency.
    import pyab3p  # type: ignore
except ImportError:  # pragma: no cover
    pyab3p = None  # type: ignore


@dataclass(frozen=True, slots=True)
class AbbreviationCandidate:
    short_form: str
    key: str
    identifier: str


@dataclass(slots=True)
class AbbreviationLookup:
    direct_lookup: dict[str, tuple[str, str]] = field(default_factory=dict)
    ambiguous_candidates: dict[str, list[AbbreviationCandidate]] = field(default_factory=dict)
    all_keys: list[str] = field(default_factory=list)
    key_to_candidates: dict[str, list[AbbreviationCandidate]] = field(default_factory=dict)
    row_counts: Counter = field(default_factory=Counter)

    def __bool__(self) -> bool:
        return bool(self.key_to_candidates)


def normalize_abbreviation_key(text: str) -> str:
    """Normalize a short form for dictionary lookup, e.g. 'S-MCs' -> 'SMCs'."""
    value = str(text).strip()
    value = re.sub(DASH_PATTERN, "", value)
    value = re.sub(r"\s+", "", value)
    return value


def abbreviation_variant_keys(text: str) -> list[str]:
    """Return singular/plural variants for short-form lookup."""
    key = normalize_abbreviation_key(text)
    variants: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        if value and value not in seen:
            variants.append(value)
            seen.add(value)

    add(key)
    if key.endswith("s") and len(key) > 1:
        add(key[:-1])
    else:
        add(key + "s")
    return variants


def abbreviation_threshold_for(mention_text: str) -> float:
    """Similarity threshold for abbreviation dictionary matching."""
    n_chars = len(normalize_abbreviation_key(mention_text))
    if n_chars <= 4:
        return 1.0
    if n_chars <= 7:
        return 0.95
    return 0.90


def ab3p_fuzzy_threshold_for(short_form_text: str) -> float:
    """Similarity threshold for matching pyab3p short forms."""
    n_chars = len(normalize_abbreviation_key(short_form_text))
    if n_chars <= 4:
        return 1.0
    if n_chars <= 7:
        return 0.95
    return 0.90


def abbreviation_sequence_ratio(left: str, right: str) -> float:
    left_norm = normalize_abbreviation_key(left)
    right_norm = normalize_abbreviation_key(right)
    if not left_norm or not right_norm:
        return 0.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def is_abbreviation_like(text: str) -> bool:
    """Heuristic used before attempting abbreviation-specific normalization."""
    raw = str(text).strip()
    if not raw:
        return False
    compact = normalize_abbreviation_key(raw)
    if len(compact) <= 1:
        return False

    has_upper = any(ch.isupper() for ch in raw)
    has_digit = any(ch.isdigit() for ch in raw)
    has_symbol = any(ch in "+/-" for ch in raw)
    no_space_shortish = len(compact) <= 12 and " " not in raw
    return has_upper or has_digit or has_symbol or no_space_shortish


def classify_abbreviation_path(abbr_path: str | Path | None) -> Optional[str]:
    """Return the recognized abbreviation-file type, or None if unavailable."""
    if abbr_path in [None, "", "."]:
        return None
    path = Path(abbr_path)
    if not path.is_file():
        return None
    if path.suffix != ".tsv":
        return "other"

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            fields = line.split("\t")
            if fields[:2] == ABBREVIATION_HEADER:
                return "short_form_identifier_tsv"
            return "other"
    return "other"


def load_abbreviation_identifier_lookup(
    abbr_paths: str | Path | Iterable[str | Path] | None,
    *,
    verbose: bool = True,
) -> AbbreviationLookup:
    """
    Load short-form -> Cell Ontology ID mappings.

    Expected TSV header:

        short_form<TAB>matched_cl_id
    """
    if abbr_paths in [None, "", "."]:
        return AbbreviationLookup()

    if isinstance(abbr_paths, (str, Path)):
        paths = [abbr_paths]
    else:
        paths = list(abbr_paths)

    key_to_candidates: dict[str, list[AbbreviationCandidate]] = defaultdict(list)
    row_counts: Counter = Counter()

    for abbr_path in paths:
        if classify_abbreviation_path(abbr_path) != "short_form_identifier_tsv":
            continue

        path = Path(abbr_path)
        if verbose:
            print(f"Loading abbreviation lookup from {path}")

        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle):
                line = line.rstrip("\n")
                if not line:
                    continue
                fields = line.split("\t")
                if line_no == 0 and fields[:2] == ABBREVIATION_HEADER:
                    continue
                if len(fields) < 2:
                    continue

                short_form, matched_cl_id = fields[:2]
                short_form = short_form.strip()
                matched_cl_id = matched_cl_id.strip()
                if not short_form or matched_cl_id in ["", "-", "None", "none"]:
                    continue

                first_id = re.split(r"[,;]", matched_cl_id)[0].strip()
                if not first_id or first_id in ["-", "None", "none"]:
                    continue

                key = normalize_abbreviation_key(short_form)
                if not key:
                    continue

                candidate = AbbreviationCandidate(
                    short_form=short_form,
                    key=key,
                    identifier=first_id,
                )
                key_to_candidates[key].append(candidate)
                row_counts[key] += 1

    direct_lookup: dict[str, tuple[str, str]] = {}
    ambiguous_candidates: dict[str, list[AbbreviationCandidate]] = {}

    for key, candidates in key_to_candidates.items():
        unique_ids = {candidate.identifier for candidate in candidates}
        if len(unique_ids) == 1:
            best = max(candidates, key=lambda item: len(item.short_form))
            direct_lookup[key] = (best.short_form, best.identifier)
        else:
            ambiguous_candidates[key] = list(candidates)

    lookup = AbbreviationLookup(
        direct_lookup=direct_lookup,
        ambiguous_candidates=ambiguous_candidates,
        all_keys=sorted(key_to_candidates.keys()),
        key_to_candidates=dict(key_to_candidates),
        row_counts=row_counts,
    )

    if verbose and lookup:
        print(
            "Loaded {} safe abbreviation mappings; {} ambiguous abbreviation keys".format(
                len(lookup.direct_lookup), len(lookup.ambiguous_candidates)
            )
        )

    return lookup


def normalize_pyab3p_output(results: Any) -> list[tuple[str, str]]:
    """Normalize pyab3p outputs across package versions."""
    if isinstance(results, dict):
        return [
            (str(short).strip(), str(long).strip())
            for short, long in results.items()
            if str(short).strip() and str(long).strip()
        ]

    if isinstance(results, list):
        pairs: list[tuple[str, str]] = []
        for item in results:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                short_form = str(item[0]).strip()
                long_form = str(item[1]).strip()
            elif isinstance(item, dict):
                short_form = str(
                    item.get("short_form")
                    or item.get("short")
                    or item.get("abbr")
                    or item.get("abbreviation")
                    or ""
                ).strip()
                long_form = str(
                    item.get("long_form")
                    or item.get("long")
                    or item.get("expansion")
                    or ""
                ).strip()
            else:
                short_form = str(
                    getattr(item, "short_form", getattr(item, "short", ""))
                ).strip()
                long_form = str(
                    getattr(item, "long_form", getattr(item, "long", ""))
                ).strip()

            if short_form and long_form:
                pairs.append((short_form, long_form))
        return pairs

    return []


def build_document_abbreviation_lookup(
    document_text_by_key: dict[str, str],
    target_document_keys: Iterable[str],
    *,
    verbose: bool = True,
) -> dict[str, dict[str, str]]:
    """
    Run Ab3P over documents and return document_key -> short_form_key -> long_form.
    """
    keys = sorted(set(str(key) for key in target_document_keys))
    if not keys:
        return {}

    if pyab3p is None:
        if verbose:
            print("pyab3p is not installed; skipping document-context abbreviation expansion.")
        return {}

    detector = pyab3p.Ab3p()
    lookup: dict[str, dict[str, str]] = {}

    for document_key in keys:
        document_text = document_text_by_key.get(document_key, "")
        if not document_text.strip():
            continue
        try:
            results = detector.get_abbrs(document_text)
        except Exception as exc:  # pragma: no cover - external package behavior
            if verbose:
                print(f"WARN: Ab3P failed for document {document_key}: {exc}")
            continue

        pairs = normalize_pyab3p_output(results)
        if not pairs:
            continue

        lookup[document_key] = {
            normalize_abbreviation_key(short_form): long_form.strip()
            for short_form, long_form in pairs
            if short_form and long_form
        }

    if verbose:
        print(f"Built Ab3P document lookup for {len(lookup)} documents")

    return lookup


def find_ab3p_long_form_with_fallback(
    doc_lookup: dict[str, str],
    matched_key: str,
) -> tuple[Optional[str], Optional[str], Optional[str], Optional[float]]:
    """
    Find a long form for a matched abbreviation key in one document lookup.

    Returns
    -------
    long_form, method, matched_doc_key, score
    """
    if not doc_lookup:
        return None, None, None, None

    exact_hit = doc_lookup.get(matched_key)
    if exact_hit:
        return exact_hit, "ab3p_exact_key", matched_key, 1.0

    for variant_key in abbreviation_variant_keys(matched_key):
        if variant_key == matched_key:
            continue
        variant_hit = doc_lookup.get(variant_key)
        if variant_hit:
            return variant_hit, "ab3p_exact_variant", variant_key, 1.0

    threshold = ab3p_fuzzy_threshold_for(matched_key)
    best_key: Optional[str] = None
    best_score = -1.0
    for doc_key in doc_lookup:
        score = abbreviation_sequence_ratio(matched_key, doc_key)
        if score > best_score:
            best_score = score
            best_key = doc_key

    if best_key is not None and best_score >= threshold:
        return doc_lookup[best_key], "ab3p_fuzzy_shortform", best_key, float(best_score)

    return None, None, None, None
