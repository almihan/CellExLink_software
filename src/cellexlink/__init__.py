"""CellExLink public API.

CellExLink provides three user-facing tasks:

1. Cell-type named entity recognition (NER).
2. Cell Ontology named entity normalization (NEN).
3. End-to-end extraction, combining NER and NEN.
"""

from __future__ import annotations

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version  # type: ignore

from .pipeline import CellExLinkPipeline, ExtractionResult, MentionInput, write_predictions_jsonl

try:
    __version__ = version("cellexlink")
except PackageNotFoundError:  # Used when running directly from source before installation.
    __version__ = "0.1.0"

__all__ = [
    "CellExLinkPipeline",
    "ExtractionResult",
    "MentionInput",
    "write_predictions_jsonl",
    "__version__",
]
