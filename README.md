# CellExLink

CellExLink is a biomedical text-mining package for **cell-type named entity recognition (NER)**, **Cell Ontology named entity normalization (NEN)**, and **end-to-end cell-type extraction** from biomedical text.

The package provides:

- a Python API for NER, NEN, and end-to-end extraction;
- BioC XML input/output utilities for benchmark and corpus workflows;
- command-line tools for model download, text prediction, BioC prediction, and BioC normalization;
- benchmark scripts for reproducing NER, gold-span NEN, and strict end-to-end results.

CellExLink combines a Bioformer-based cell-type recognizer with a SapBERT-based Cell Ontology normalizer. The normalization step includes Cell Ontology alias retrieval, abbreviation handling, Ab3P-based document-level long-form recovery, and concept reranking.

---

## Repository layout

```text
CellExLink/
├── README.md
├── LICENSE.txt
├── CITATION.cff
├── pyproject.toml
├── MANIFEST.in
├── src/
│   └── cellexlink/
│       ├── __init__.py
│       ├── pipeline.py
│       ├── cli.py
│       ├── io.py
│       ├── recognition/
│       ├── normalization/
│       └── resources/
├── examples/
├── benchmarks/
└── tests/
```

Large model checkpoints and benchmark datasets are **not** stored in this repository. Download them separately and place them under `models/` and `data/` or pass their paths explicitly.

---

## Installation

### Install from GitHub

```bash
git clone https://github.com/almihan/CellExLink_software.git
cd CellExLink_software

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The base installation includes Ab3P support through `pyab3p`, so abbreviation expansion is available without installing a separate optional extra.

### Development installation

```bash
python -m pip install -e ".[dev]"
pytest -q
```

### Benchmark dependencies

```bash
python -m pip install -e ".[benchmarks]"
```

---

## Model checkpoints

Download or place the CellExLink model checkpoints locally, for example:

```bash
cellexlink download-models --output-dir models
```

Expected local model paths:

```text
models/CellExLink-bioformer16L
models/CellExLink-Sapbert
```

If your models are stored elsewhere, pass the paths with `ner_model`, `nen_model`, `--ner-model`, or `--nen-model`.

---

## Python API quick start

```python
from cellexlink import CellExLinkPipeline

text = "The mesothelial cell and SMC clusters formed the third population."

pipe = CellExLinkPipeline.from_pretrained(
    ner_model="models/CellExLink-bioformer16L",
    nen_model="models/CellExLink-Sapbert",
    ontology_path="src/cellexlink/resources/cell_ontology_v2025-12-17.jsonl",
    abbreviations_path="src/cellexlink/resources/abbreviations.tsv",
)
```

### 1. NER only

```python
ner_results = pipe.recognize_text(text)

for result in ner_results:
    print(result.to_dict())
```

### 2. NEN only

Use this when mentions are already known, for example from gold annotations or another recognizer.

```python
nen_results = pipe.normalize_mentions(
    mentions=[
        {"text": "mesothelial cell", "start": text.index("mesothelial cell")},
        {"text": "SMC", "start": text.index("SMC")},
    ],
    document_text=text,
)

for result in nen_results:
    print(result.to_dict())
```

### 3. End-to-end extraction

```python
e2e_results = pipe.extract_text(text)

for result in e2e_results:
    print(result.to_dict())
```

---

## BioC XML API

### NER-only BioC prediction

```python
pipe.recognize_bioc(
    input_xml="examples/sample_input.xml",
    output_xml="outputs/sample.ner.xml",
)
```

### Gold-span NEN BioC prediction

```python
pipe.normalize_bioc(
    input_xml="examples/sample_gold_spans.xml",
    output_xml="outputs/sample.normalized.xml",
)
```

### End-to-end BioC extraction

```python
pipe.extract_bioc(
    input_xml="examples/sample_input.xml",
    output_xml="outputs/sample.end_to_end.xml",
    ner_output_xml="outputs/sample.ner.xml",
)
```

A complete BioC example is available in:

```bash
python examples/quickstart_bioc.py
```

A plain Python API example is available in:

```bash
python examples/quickstart_api.py
```

---

## Command-line usage

Check available options with:

```bash
cellexlink --help
cellexlink predict-text --help
cellexlink predict-bioc --help
cellexlink normalize-bioc --help
```

### End-to-end prediction from plain text

```bash
cellexlink predict-text \
  --text "The mesothelial cell and SMC clusters formed the third population." \
  --output outputs/text_predictions.jsonl \
  --ner-model models/CellExLink-bioformer16L \
  --nen-model models/CellExLink-Sapbert
```

### End-to-end prediction from BioC XML

```bash
cellexlink predict-bioc \
  --input examples/sample_input.xml \
  --output outputs/sample.end_to_end.xml \
  --ner-output outputs/sample.ner.xml \
  --ner-model models/CellExLink-bioformer16L \
  --nen-model models/CellExLink-Sapbert \
  --ontology-path src/cellexlink/resources/cell_ontology_v2025-12-17.jsonl \
  --abbreviations-path src/cellexlink/resources/abbreviations.tsv
```

### Gold-span NEN from BioC XML

```bash
cellexlink normalize-bioc \
  --input examples/sample_gold_spans.xml \
  --output outputs/sample.normalized.xml \
  --nen-model models/CellExLink-Sapbert \
  --ontology-path src/cellexlink/resources/cell_ontology_v2025-12-17.jsonl \
  --abbreviations-path src/cellexlink/resources/abbreviations.tsv
```

---

## Input and output formats

CellExLink supports BioC XML for corpus and benchmark workflows. NER predictions are written as BioC annotations with mention spans. NEN and end-to-end outputs add Cell Ontology prediction fields such as:

```text
CellExLink-Sapbert_id_0
CellExLink-Sapbert_identifier_name_0
CellExLink-Sapbert_identifier_score_0
CellExLink-Sapbert_match_source
```

The plain-text CLI writes JSON Lines, where each line is one extracted mention and optional linked Cell Ontology identifier.

---

## Benchmarks

The benchmark folder contains minimal scripts for the three evaluation settings:

```text
benchmarks/run_cellexlink.py
benchmarks/evaluate_ner.py
benchmarks/evaluate_nen.py
benchmarks/evaluate_end_to_end.py
```

Use `run_cellexlink.py` first to generate prediction files, then run the corresponding evaluation script.

Set paths as needed:

```bash
DATA=data/evaluation
MODELS=models
OUT=benchmark_outputs/cellexlink
```

### 1. Generate NER predictions

```bash
python benchmarks/run_cellexlink.py \
  --mode ner \
  --input $DATA/CellLink/validation.xml \
  --input $DATA/CRAFT/test.xml \
  --input $DATA/BioID/test.xml \
  --input $DATA/AnatEM/test.xml \
  --input $DATA/JNLPBA/test.xml \
  --output-dir $OUT/ner \
  --ner-model $MODELS/CellExLink-bioformer16L \
  --batch-size 16
```

### 2. Generate gold-span NEN predictions

Gold-span NEN uses the gold entity spans in the input BioC XML and predicts Cell Ontology identifiers for those spans.

```bash
python benchmarks/run_cellexlink.py \
  --mode normalize \
  --input $DATA/CellLink/validation.xml \
  --input $DATA/CRAFT/test.xml \
  --input $DATA/BioID/test.xml \
  --output-dir $OUT/nen_gold \
  --nen-model $MODELS/CellExLink-Sapbert \
  --ontology-path src/cellexlink/resources/cell_ontology_v2025-12-17.jsonl \
  --abbreviations-path src/cellexlink/resources/abbreviations.tsv \
  --batch-size 16
```

Do not remove the gold `identifier` infons for this evaluation. The normalizer does not use those gold identifiers, but the evaluator needs them to compare with the predicted `CellExLink-Sapbert_id_0` fields.

### 3. Generate strict end-to-end predictions

Strict end-to-end evaluation requires both the mention span and Cell Ontology identifier to be correct.

```bash
python benchmarks/run_cellexlink.py \
  --mode full \
  --input $DATA/CellLink/validation.xml \
  --input $DATA/CRAFT/test.xml \
  --input $DATA/BioID/test.xml \
  --output-dir $OUT/end_to_end \
  --ner-model $MODELS/CellExLink-bioformer16L \
  --nen-model $MODELS/CellExLink-Sapbert \
  --ontology-path src/cellexlink/resources/cell_ontology_v2025-12-17.jsonl \
  --abbreviations-path src/cellexlink/resources/abbreviations.tsv \
  --batch-size 16
```

### 4. Evaluate

NER:

```bash
python benchmarks/evaluate_ner.py \
  --gold CellLink=$DATA/CellLink/validation.xml \
  --pred CellLink=$OUT/ner/CellLink_validation.ner.xml \
  --gold CRAFT=$DATA/CRAFT/test.xml \
  --pred CRAFT=$OUT/ner/CRAFT_test.ner.xml \
  --gold BioID=$DATA/BioID/test.xml \
  --pred BioID=$OUT/ner/BioID_test.ner.xml \
  --gold AnatEM=$DATA/AnatEM/test.xml \
  --pred AnatEM=$OUT/ner/AnatEM_test.ner.xml \
  --gold JNLPBA=$DATA/JNLPBA/test.xml \
  --pred JNLPBA=$OUT/ner/JNLPBA_test.ner.xml \
  --macro-average \
  --output-csv benchmark_outputs/table_ner_results.csv
```

Gold-span NEN:

```bash
python benchmarks/evaluate_nen.py \
  --dataset-style other \
  --gold CRAFT=$DATA/CRAFT/test.xml \
  --pred CRAFT=$OUT/nen_gold/CRAFT_test.normalized.xml \
  --gold BioID=$DATA/BioID/test.xml \
  --pred BioID=$OUT/nen_gold/BioID_test.normalized.xml \
  --model-names CellExLink-Sapbert \
  --output-csv benchmark_outputs/table_nen_results.csv
```

For CellLink, run the CellLink-specific setting separately:

```bash
python benchmarks/evaluate_nen.py \
  --dataset-style celllink \
  --gold CellLink=$DATA/CellLink/validation.xml \
  --pred CellLink=$OUT/nen_gold/CellLink_validation.normalized.xml \
  --model-names CellExLink-Sapbert \
  --output-csv benchmark_outputs/table_nen_celllink_results.csv
```

Strict end-to-end:

```bash
python benchmarks/evaluate_end_to_end.py \
  --dataset-style other \
  --gold CRAFT=$DATA/CRAFT/test.xml \
  --pred CRAFT=$OUT/end_to_end/CRAFT_test.normalized.xml \
  --gold BioID=$DATA/BioID/test.xml \
  --pred BioID=$OUT/end_to_end/BioID_test.normalized.xml \
  --model-names CellExLink-Sapbert \
  --output-csv benchmark_outputs/table_end_to_end_results.csv
```

For CellLink:

```bash
python benchmarks/evaluate_end_to_end.py \
  --dataset-style celllink \
  --gold CellLink=$DATA/CellLink/validation.xml \
  --pred CellLink=$OUT/end_to_end/CellLink_validation.normalized.xml \
  --model-names CellExLink-Sapbert \
  --output-csv benchmark_outputs/table_end_to_end_celllink_results.csv
```

---

## Tests

The default test suite is lightweight and does not require downloading the large model checkpoints.

```bash
pytest -q
```

Before a release, also check packaging:

```bash
python -m build
python -m twine check dist/*
```

---

## Data availability

Benchmark datasets are available from Zenodo:

```text
https://zenodo.org/records/18090009
```

Model checkpoint download instructions are provided through the repository documentation and the `cellexlink download-models` command.

---

## Citation

If you use CellExLink, please cite the CellExLink software paper and the repository release.

```bibtex
@software{cellexlink,
  title = {CellExLink: End-to-End Cell-Type Recognition and Cell Ontology Normalization from Biomedical Text},
  author = {CellExLink contributors},
  year = {2026},
  url = {https://github.com/almihan/CellExLink_software}
}
```

A `CITATION.cff` file is included for GitHub citation support. Update it with the final SoftwareX DOI after acceptance.

---

## License

CellExLink is distributed under the Apache License, Version 2.0. See `LICENSE.txt`.

Third-party datasets, model checkpoints, ontologies, and dependencies may have separate licenses or usage terms. Users are responsible for complying with those terms.
