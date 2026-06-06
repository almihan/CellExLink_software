"""
Recognition module for CellExLink.

This subpackage contains the cell-type named entity recognition (NER)
component used before Cell Ontology normalization.
"""

from __future__ import annotations

from .bioc import EntitySpan, PassageRecord, convert_bioc_to_json, write_predictions_to_bioc_xml
from .predict import predict_ner
from .train import train_ner

__all__ = [
    "EntitySpan",
    "PassageRecord",
    "convert_bioc_to_json",
    "write_predictions_to_bioc_xml",
    "predict_ner",
    "train_ner",
]
