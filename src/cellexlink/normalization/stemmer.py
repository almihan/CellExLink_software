"""Original-compatible plural normalization for CellExLink NEN.

This module intentionally keeps the same conservative behavior as the
original CellExLink ``normalization/plural_stemmer.py``.  Do not replace this
with lowercasing, Unicode normalization, or a general-purpose stemmer, because
those changes alter the strings embedded by SapBERT and can change NEN output.
"""

from __future__ import annotations

import re
from typing import Iterable

# Same token pattern as the original code: words, punctuation, underscore,
# comma.  This means strings such as ``CD8+ T cells`` become
# ``CD8 + T cell`` after plural normalization.
_TOKEN_FINDER = re.compile(r"[^\W_]+|[^\w\s]|_|,", re.UNICODE)


def split_tokens(value: object) -> list[str]:
    """Split text into the original CellExLink token representation."""

    return _TOKEN_FINDER.findall(str(value))


def _replace_tail(word: str, suffix: str, replacement: str) -> str:
    return word[: -len(suffix)] + replacement


def normalize_token(value: object) -> str:
    """Conservatively singularize one token.

    This is intentionally not a linguistic stemmer.  It only applies the small
    set of suffix rules used by the original CellExLink normalizer.
    """

    word = str(value)
    if not word.endswith("s"):
        return word

    if word.endswith("viruses"):
        return _replace_tail(word, "uses", "us")

    if word.endswith("ies"):
        if not word.endswith(("eies", "aies")):
            return _replace_tail(word, "ies", "y")

    if word.endswith("es"):
        if not word.endswith(("aes", "ees", "oes")):
            if word.endswith("sses"):
                return _replace_tail(word, "es", "")
            return _replace_tail(word, "es", "e")

    if word.endswith(("us", "ss")):
        return word

    return _replace_tail(word, "s", "")


def normalize_text(value: object) -> str:
    """Normalize a mention or ontology alias exactly as the original did."""

    return " ".join(normalize_token(part) for part in split_tokens(value))


# Compatibility name used by the linker and older code.
plural_normalize_text = normalize_text


__all__ = [
    "split_tokens",
    "normalize_token",
    "normalize_text",
    "plural_normalize_text",
]
