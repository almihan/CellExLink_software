"""Abbreviation handling for CellExLink normalization.

This keeps the original CellExLink strategy:
1. Load a TSV dictionary with short_form and matched_cl_id.
2. Directly assign unambiguous abbreviation keys.
3. Treat keys with multiple CL IDs as ambiguous.
4. For ambiguous keys, use document-level Ab3P long-form recovery when
   available, then pass the long form through the same ontology linker.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, Optional

try:  # Optional dependency; tests should pass without it.
    import pyab3p  # type: ignore
except ImportError:  # pragma: no cover
    pyab3p = None  # type: ignore

DEFAULT_ABBREVIATIONS_FILENAME = "abbreviations.tsv"
ABBREVIATION_HEADER = ["short_form", "matched_cl_id"]
DASH_PATTERN = r"[\-\u2010\u2011\u2012\u2013\u2014\u2212]"


@dataclass(slots=True)
class AbbreviationCandidate:
    short_form: str
    key: str
    identifier: str


@dataclass(slots=True)
class AbbreviationLookup:
    direct_lookup: dict[str, tuple[str, str]] = field(default_factory=dict)
    ambiguous_candidates: dict[str, list[AbbreviationCandidate]] = field(
        default_factory=dict
    )
    all_keys: list[str] = field(default_factory=list)
    key_to_candidates: dict[str, list[AbbreviationCandidate]] = field(
        default_factory=dict
    )
    row_counts: Counter = field(default_factory=Counter)

    def __bool__(self) -> bool:
        return bool(self.direct_lookup or self.ambiguous_candidates)


def default_abbreviations_path() -> Path:
    candidate = resources.files("cellexlink").joinpath(
        "resources", DEFAULT_ABBREVIATIONS_FILENAME
    )
    return Path(str(candidate))


def normalize_abbreviation_key(text: object) -> str:
    """Normalize abbreviation keys exactly as the original code did."""

    value = str(text).strip()
    value = re.sub(DASH_PATTERN, "", value)
    value = re.sub(r"\s+", "", value)
    return value


def abbreviation_variant_keys(text: object) -> list[str]:
    key = normalize_abbreviation_key(text)
    variants: list[str] = []
    seen: set[str] = set()

    def add(item: str) -> None:
        if item and item not in seen:
            seen.add(item)
            variants.append(item)

    add(key)
    if key.endswith("s") and len(key) > 1:
        add(key[:-1])
    else:
        add(key + "s")
    return variants


def abbreviation_threshold_for(mention_text: object) -> float:
    n = len(normalize_abbreviation_key(mention_text))
    if n <= 4:
        return 1.0
    if n <= 7:
        return 0.95
    return 0.90


def ab3p_fuzzy_threshold_for(short_form_text: object) -> float:
    n = len(normalize_abbreviation_key(short_form_text))
    if n <= 4:
        return 1.0
    if n <= 7:
        return 0.95
    return 0.90


def abbreviation_sequence_ratio(left: object, right: object) -> float:
    from difflib import SequenceMatcher

    left_norm = normalize_abbreviation_key(left)
    right_norm = normalize_abbreviation_key(right)
    if not left_norm or not right_norm:
        return 0.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def is_abbreviation_like(text: object) -> bool:
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


def _normalize_paths(
    paths: str | Path | Iterable[str | Path] | None,
) -> list[Path]:
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(item) for item in paths]


def load_abbreviation_identifier_lookup(
    abbr_paths: str | Path | Iterable[str | Path] | None,
    *,
    verbose: bool = True,
) -> AbbreviationLookup:
    """Load short-form-to-CL-ID mappings from TSV files.

    Keys with exactly one unique CL ID go into ``direct_lookup``.  Keys with
    multiple IDs remain in ``ambiguous_candidates`` so Ab3P/document context can
    decide whether a long form is available.
    """

    key_to_candidates: dict[str, list[AbbreviationCandidate]] = defaultdict(list)
    row_counts: Counter = Counter()

    for path in _normalize_paths(abbr_paths):
        if classify_abbreviation_path(path) != "short_form_identifier_tsv":
            continue

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
                if not short_form:
                    continue
                if matched_cl_id in ["", "-", "None", "none"]:
                    continue

                first_id = re.split(r"[,;]", matched_cl_id)[0].strip()
                if first_id in ["", "-", "None", "none"]:
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
    """Run Ab3P on full document text and return doc -> short key -> long form."""

    keys = sorted({str(key) for key in target_document_keys})
    if not keys:
        return {}

    if pyab3p is None:  # pragma: no cover - optional dependency
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
        except Exception as exc:  # pragma: no cover - defensive around external lib
            if verbose:
                print(f"WARN Ab3P failed for document {document_key}: {exc}")
            continue

        pairs = normalize_pyab3p_output(results)
        if not pairs:
            continue

        doc_lookup: dict[str, str] = {}
        for short_form, long_form in pairs:
            if not short_form or not long_form:
                continue
            key = normalize_abbreviation_key(short_form)
            if key:
                doc_lookup[key] = long_form.strip()
        if doc_lookup:
            lookup[document_key] = doc_lookup

    if verbose:
        print(f"Built Ab3P document lookup for {len(lookup)} documents")

    return lookup


__all__ = [
    "DEFAULT_ABBREVIATIONS_FILENAME",
    "ABBREVIATION_HEADER",
    "AbbreviationCandidate",
    "AbbreviationLookup",
    "default_abbreviations_path",
    "normalize_abbreviation_key",
    "abbreviation_variant_keys",
    "abbreviation_threshold_for",
    "ab3p_fuzzy_threshold_for",
    "abbreviation_sequence_ratio",
    "is_abbreviation_like",
    "classify_abbreviation_path",
    "load_abbreviation_identifier_lookup",
    "normalize_pyab3p_output",
    "build_document_abbreviation_lookup",
]
