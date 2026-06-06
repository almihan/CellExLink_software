# CellExLink

CellExLink is a biomedical natural language processing package for end-to-end **cell-type recognition** and **Cell Ontology normalization** from biomedical text.

The package provides two connected stages:

1. **Recognition**: detect cell-type mentions in text.
2. **Normalization**: link recognized mentions to Cell Ontology identifiers.

CellExLink supports both plain-text input and BioC XML input. The recommended user-facing interfaces are the Python API and the `cellexlink` command-line tool.

---

## Repository structure

```text
CellExLink/
├── src/cellexlink/          # installable Python package
├── examples/                # small runnable examples
├── tests/                   # model-free unit and smoke tests
├── docs/                    # installation, usage, I/O, models, benchmarks, troubleshooting
├── docker/                  # optional Docker support
├── scripts/                 # model, dataset, and ontology-resource utilities
├── benchmarks/              # optional paper-result reproducibility material
├── pyproject.toml
├── README.md
├── LICENSE.txt
├── CITATION.cff
├── MANIFEST.in
├── environment.yml
├── requirements.txt
├── requirements-dev.txt
├── requirements-benchmarks.txt
└── .gitignore
```

The core software is under `src/cellexlink/`. Benchmarking and baseline-comparison code should remain outside the installable package.

---

## Installation

### Option 1: install from a local clone

```bash
git clone https://github.com/ShahriyariLab/CellExLink-End-to-End-Cell-Type-Extraction-and-Cell-Ontology-Normalization-from-Biomedical-Text.git
cd CellExLink-End-to-End-Cell-Type-Extraction-and-Cell-Ontology-Normalization-from-Biomedical-Text

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

For full abbreviation expansion support with Ab3P:

```bash
pip install -e ".[abbr]"
```

For development and tests:

```bash
pip install -e ".[dev]"
pytest -q
```

For benchmark scripts:

```bash
pip install -e ".[benchmarks]"
```

### Option 2: create a Conda environment

```bash
conda env create -f environment.yml
conda activate cellexlink
```

---

## Download model checkpoints

CellExLink does not store large model files in the GitHub repository. Download the default checkpoints with:

```bash
cellexlink download-models --output-dir models
```

or with the script:

```bash
python scripts/download_models.py --output-dir models
```

This creates a local directory similar to:

```text
models/
├── CellExLink-bioformer16L/
├── CellExLink-Sapbert/
└── models_manifest.json
```

You can then use local paths:

```bash
cellexlink predict-bioc \
  --input examples/sample_input.xml \
  --output outputs/sample_output.normalized.xml \
  --ner-model models/CellExLink-bioformer16L \
  --nen-model models/CellExLink-Sapbert
```

---

## Quick start: plain text

```bash
python examples/quickstart_text.py --print-results
```

Using local models:

```bash
python examples/quickstart_text.py \
  --ner-model models/CellExLink-bioformer16L \
  --nen-model models/CellExLink-Sapbert \
  --print-results
```

Python API:

```python
from cellexlink import CellExLinkPipeline

pipe = CellExLinkPipeline.from_pretrained(
    ner_model="models/CellExLink-bioformer16L",
    nen_model="models/CellExLink-Sapbert",
)

results = pipe.extract_text(
    "The mesothelial cell and SMC clusters formed the third population."
)

for result in results:
    print(result.to_dict())
```

Command-line API:

```bash
cellexlink predict-text \
  --input examples/sample_input.txt \
  --output outputs/text_predictions.jsonl \
  --ner-model models/CellExLink-bioformer16L \
  --nen-model models/CellExLink-Sapbert
```

---

## Quick start: BioC XML

```bash
python examples/quickstart_bioc.py --print-results
```

Using local models:

```bash
python examples/quickstart_bioc.py \
  --ner-model models/CellExLink-bioformer16L \
  --nen-model models/CellExLink-Sapbert \
  --print-results
```

Command-line API:

```bash
cellexlink predict-bioc \
  --input examples/sample_input.xml \
  --output outputs/sample_output.normalized.xml \
  --ner-output outputs/sample_output.ner.xml \
  --ner-model models/CellExLink-bioformer16L \
  --nen-model models/CellExLink-Sapbert
```

---

## Recognition-only and normalization-only modes

Run only the NER stage:

```bash
python -m cellexlink.recognition.predict \
  --model-path models/CellExLink-bioformer16L \
  --input-xml examples/sample_input.xml \
  --output-dir outputs/ner \
  --output-xml outputs/ner_predictions.xml
```

Run only the normalization stage on BioC XML that already contains cell-type annotations:

```bash
cellexlink normalize-bioc \
  --input outputs/ner_predictions.xml \
  --output outputs/normalized.xml \
  --nen-model models/CellExLink-Sapbert
```

---

## Input and output formats

CellExLink supports:

- plain text files (`.txt`),
- BioC XML files (`.xml`),
- JSONL passage files for intermediate recognition data,
- normalized BioC XML output,
- JSONL summaries of extracted mentions and linked Cell Ontology identifiers.

See:

```text
docs/input_output.md
```

---

## Resources

Packaged resources are stored in:

```text
src/cellexlink/resources/
├── abbreviations.tsv
└── cell_ontology_v2025-12-17.jsonl
```

Use the ontology-building script when the Cell Ontology resource needs to be regenerated:

```bash
python scripts/build_cell_ontology_resource.py \
  --input-obo data/raw/cl_2025-12-17.obo \
  --output-jsonl src/cellexlink/resources/cell_ontology_v2025-12-17.jsonl \
  --validate-with-package
```

---

## Datasets

Datasets should not be bundled inside the Python package. Download them separately:

```bash
python scripts/download_datasets.py --output-dir data
```

The script writes a dataset manifest and a short README into the output directory. Review the original dataset licenses and terms of use before redistribution.

---

## Benchmarks

Benchmarking and comparison with external tools should be kept in:

```text
benchmarks/
```

The installable package should not depend on baseline tools such as ScispaCy, BERN2, or VANER2. Use separate benchmark instructions and optional environments for those tools.

A typical benchmark workflow is:

```bash
python scripts/download_models.py --output-dir models
python scripts/download_datasets.py --output-dir data
pip install -r requirements-benchmarks.txt
python benchmarks/run_cellexlink.py --data-dir data --model-dir models --output-dir benchmarks/results
python benchmarks/make_tables.py --results-dir benchmarks/results
```

---

## Tests

The default tests are model-free and should run quickly on CPU:

```bash
pip install -e ".[dev]"
pytest -q
```

Slow tests that download or load real model checkpoints should be marked with:

```python
@pytest.mark.slow
```

and run separately:

```bash
pytest -q -m slow
```

---

## Docker

Build from the repository root:

```bash
docker build -f docker/Dockerfile -t cellexlink:latest .
```

Run the model-free Docker smoke test:

```bash
docker run --rm cellexlink:latest python /app/docker/docker_test.py
```

Run with mounted local models:

```bash
docker run --rm \
  -v "$PWD/models:/models" \
  -v "$PWD/examples:/workspace/examples" \
  -v "$PWD/outputs:/workspace/outputs" \
  cellexlink:latest \
  cellexlink predict-bioc \
    --input /workspace/examples/sample_input.xml \
    --output /workspace/outputs/sample_output.normalized.xml \
    --ner-model /models/CellExLink-bioformer16L \
    --nen-model /models/CellExLink-Sapbert
```

---

## Citation

If you use CellExLink, please cite the SoftwareX paper and the software release. Update `CITATION.cff` with the final author list, DOI, and release date before submission.

```bibtex
@software{cellexlink,
  title = {CellExLink: End-to-End Cell-Type Recognition and Normalization in Biomedical Text},
  author = {{CellExLink contributors}},
  year = {2026},
  url = {https://github.com/ShahriyariLab/CellExLink-End-to-End-Cell-Type-Extraction-and-Cell-Ontology-Normalization-from-Biomedical-Text}
}
```

---

## License

This project is distributed under the MIT License. See `LICENSE.txt`.

Check the licenses of external datasets, pretrained models, ontology files, and third-party tools before redistribution.
