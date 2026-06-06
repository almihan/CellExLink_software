# VANER2 baseline

VANER2 is treated as an external NER baseline. Keep it in a separate environment or service outside the main CellExLink package.

Run VANER2 on the benchmark passages, convert predicted cell-type spans to BioC XML or CellExLink benchmark JSONL, and evaluate with `evaluate_ner.py`. Because VANER2 is a recognition baseline rather than a Cell Ontology linker, do not include it in linked extraction unless you add and document a separate CL normalization component.
