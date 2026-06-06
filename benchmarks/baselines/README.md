# Baseline tools

This folder documents optional baseline evaluation. Baseline tools are not part of the CellExLink package and should not be installed by default.

Recommended policy:

1. Keep each baseline in its own environment.
2. Save baseline raw outputs outside the repository, for example under `benchmark_outputs/baselines_raw/`.
3. Convert baseline outputs to BioC XML or CellExLink benchmark JSONL.
4. Evaluate with the shared scripts in `benchmarks/`.

Suggested output schema for converted JSONL:

```json
{"document_id":"doc1","passage_id":0,"passage_offset":0,"text":"...","entities":[{"start":10,"end":23,"text":"T cells","label":"cell_type","cl_id":"CL:0000084"}]}
```

For NER-only tools such as VANER2, omit `cl_id` and use `evaluate_ner.py`. For tools that provide linked identifiers, include `cl_id` and use `evaluate_nen.py` or `evaluate_end_to_end.py`.


# External baseline scripts

This folder contains optional baseline material for reproducing benchmark
comparisons in the CellExLink SoftwareX paper.

## Included executable baseline

- `scispacy/`: runnable ScispaCy NER and Cell Ontology linking baseline.

## Documented-only baselines

- `bern2/`: instructions and provenance notes only.
- `vaner2/`: instructions and provenance notes only.

BERN2 and VANER2 are not installed or run by the CellExLink package. Their
outputs can be evaluated with the shared benchmark scripts if users provide
BioC XML prediction files in the expected format.