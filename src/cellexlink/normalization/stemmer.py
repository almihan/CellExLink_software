"""
Conservative text normalization utilities for CellExLink normalization.

The goal here is not full linguistic stemming. For ontology linking, we only need
a stable representation that reduces common plural/singular variation in cell-type
names while avoiding aggressive biomedical term distortion.
"""

from __future__ import annotations

import re
import unicodedata

DASH_PATTERN = r"[\-\u2010\u2011\u2012\u2013\u2014\u2212]"
_WHITESPACE_RE = re.compile(r"\s+")
_EDGE_PUNCT_RE = re.compile(r"^[\s\"'`.,;:()\[\]{}]+|[\s\"'`.,;:()\[\]{}]+$")

GREEK_REPLACEMENTS = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "κ": "kappa",
    "λ": "lambda",
    "μ": "mu",
    "π": "pi",
    "τ": "tau",
    "ω": "omega",
}


IRREGULAR_PLURALS = {
    "bacteria": "bacterium",
    "criteria": "criterion",
    "data": "datum",
    "fibres": "fibre",
    "fibers": "fiber",
    "media": "medium",
    "nuclei": "nucleus",
    "phenomena": "phenomenon",
}


DO_NOT_STRIP_S = {
    "apoptosis",
    "class",
    "dendritus",
    "fibrosis",
    "gliosis",
    "homeostasis",
    "hypothalamus",
    "islets",  # often appears as a fixed biomedical phrase
    "langerhans",
    "mesenchymis",
    "nervous",
    "nucleus",
    "status",
    "thymus",
}


def normalize_unicode(text: str) -> str:
    """Normalize Unicode while preserving biomedical symbols such as + and /."""
    value = unicodedata.normalize("NFKC", str(text))
    for greek, replacement in GREEK_REPLACEMENTS.items():
        value = value.replace(greek, replacement)
        value = value.replace(greek.upper(), replacement)
    return value


def normalize_dashes(text: str, replacement: str = "-") -> str:
    """Map Unicode dash characters to a stable dash representation."""
    return re.sub(DASH_PATTERN, replacement, str(text))


def collapse_whitespace(text: str) -> str:
    """Collapse repeated whitespace and strip leading/trailing whitespace."""
    return _WHITESPACE_RE.sub(" ", str(text)).strip()


def strip_edge_punctuation(text: str) -> str:
    """Remove punctuation only at token boundaries."""
    return _EDGE_PUNCT_RE.sub("", str(text))


def singularize_token(token: str) -> str:
    """
    Conservative singularization for common cell-type mentions.

    Examples
    --------
    cells -> cell
    lymphocytes -> lymphocyte
    SMCs -> smc after case folding in normalize_text
    bodies -> body
    """
    token = strip_edge_punctuation(token.casefold())
    if not token:
        return token

    if token in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[token]

    if token in DO_NOT_STRIP_S:
        return token

    # Keep short non-biological words stable.
    if len(token) <= 3:
        return token

    # bodies -> body, colonies -> colony
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"

    # lymphocytes -> lymphocyte, macrophages -> macrophage
    if len(token) > 4 and token.endswith("es"):
        if token.endswith(("ches", "shes", "xes", "zes", "ses")):
            return token[:-2]
        if token.endswith("tes") or token.endswith("ges"):
            return token[:-1]

    # cells -> cell, SMCs -> smc, neurons -> neuron
    if len(token) > 3 and token.endswith("s"):
        if token.endswith(("ss", "us", "is")):
            return token
        return token[:-1]

    return token


def normalize_text(text: str, *, singularize: bool = True) -> str:
    """
    Normalize text for dictionary lookup and embedding queries.

    This function intentionally keeps spaces, plus signs, slashes and hyphens
    because they can be meaningful in biomedical entity names.
    """
    value = normalize_unicode(str(text))
    value = normalize_dashes(value, replacement="-")
    value = value.casefold()
    value = collapse_whitespace(value)

    if not singularize:
        return value

    tokens = [singularize_token(tok) for tok in value.split(" ")]
    return collapse_whitespace(" ".join(tok for tok in tokens if tok))


# Backward-compatible name used by the old code.
def plural_normalize_text(text: str) -> str:
    return normalize_text(text, singularize=True)
