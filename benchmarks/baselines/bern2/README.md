# BERN2 baseline

BERN2 is an external biomedical NER/NEN system. Keep BERN2 outside the CellExLink package and record the exact version, service endpoint, or Docker image used for a benchmark run.

Suggested workflow: run BERN2 on the same passages, save raw output outside the repository, convert cell-type mentions and CL identifiers to BioC XML or CellExLink benchmark JSONL, and evaluate with `evaluate_ner.py` and `evaluate_end_to_end.py`.
