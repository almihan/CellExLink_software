"""
CellExLink: End-to-end cell-type recognition and Cell Ontology normalization.

CellExLink provides:
1. Cell-type named entity recognition from biomedical text.
2. Cell Ontology normalization for detected cell-type mentions.
3. End-to-end BioC XML processing for biomedical literature-mining workflows.
"""

from __future__ import annotations

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version  # type: ignore

from .pipeline import CellExLinkPipeline, ExtractionResult

try:
    __version__ = version("cellexlink")
except PackageNotFoundError:  
    # Used when running directly from source before installation.
    __version__ = "0.1.0"

__all__ = [
    "CellExLinkPipeline",
    "ExtractionResult",
    "__version__",
]