# Installation

This guide explains how to install CellExLink for normal use, local development, testing, and benchmarking.

CellExLink has two main software components:

1. **Cell-type recognition**, which detects cell-type spans in biomedical text.
2. **Cell Ontology normalization**, which maps recognized cell-type mentions to Cell Ontology identifiers.

The recommended installation for SoftwareX review is an editable source installation from the GitHub repository, followed by a small example run.

---

## 1. Requirements

Recommended environment:

- Python 3.10, 3.11, or 3.12
- Linux or macOS for normal use
- A CPU is sufficient for testing and small examples
- A CUDA-capable GPU is recommended for large-scale inference or training

Core Python packages are declared in `pyproject.toml`. Typical core dependencies include:

- `torch`
- `transformers`
- `datasets`
- `sentence-transformers`
- `numpy`
- `scipy`
- `tqdm`
- `huggingface-hub`
- `pyab3p`, optional but recommended for abbreviation expansion

Benchmark dependencies are intentionally separated from the core package because some baseline tools require older or separate Python environments.

---

## 2. Create a clean environment

Using `conda`:

```bash
conda create -n cellexlink python=3.11 -y
conda activate cellexlink
python -m pip install --upgrade pip
```

Using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

---

## 3. Install from source

From the repository root:

```bash
git clone https://github.com/ShahriyariLab/CellExLink.git
cd CellExLink
pip install -e .
```

For development and tests:

```bash
pip install -e ".[dev]"
```

For benchmarking scripts:

```bash
pip install -e ".[dev,benchmarks]"
```

If your shell treats square brackets specially, quote the package specifier:

```bash
pip install -e '.[dev,benchmarks]'
```

---

## 4. Install from PyPI

After the package is released on PyPI, users should be able to install it with:

```bash
pip install cellexlink
```

Until the first public release is available, use the editable source installation above.

---

## 5. Verify the installation

Check that the package imports:

```bash
python - <<'PY'
import cellexlink
from cellexlink import CellExLinkPipeline
print("CellExLink version:", cellexlink.__version__)
print(CellExLinkPipeline)
PY
```

Check the command-line interface:

```bash
cellexlink --help
cellexlink predict-text --help
cellexlink predict-bioc --help
cellexlink normalize-bioc --help
```

Run the fast test suite:

```bash
pytest -q
```

The default tests are designed to be model-free and CPU-friendly. They should not download Hugging Face checkpoints.

---

## 6. Download pretrained checkpoints

CellExLink uses separate pretrained checkpoints for recognition and normalization.

Default model IDs:

```text
NER model: almire/CellExLink-bioformer16L
NEN model: almire/CellExLink-Sapbert
```

Download the default models:

```bash
cellexlink download-models --output-dir models
```

This creates a directory similar to:

```text
models/
├── CellExLink-bioformer16L/
├── CellExLink-Sapbert/
└── models.json
```

Then run examples with local model paths:

```bash
python examples/quickstart_text.py \
  --ner-model models/CellExLink-bioformer16L \
  --nen-model models/CellExLink-Sapbert \
  --print-results
```

You may also allow Hugging Face to download models automatically by using the default model IDs:

```bash
python examples/quickstart_text.py --print-results
```

---

## 7. Run the examples

Plain text example:

```bash
python examples/quickstart_text.py --print-results
```

BioC XML example:

```bash
python examples/quickstart_bioc.py --print-results
```

Command-line examples:

```bash
cellexlink predict-text \
  --input examples/sample_input.txt \
  --output outputs/sample_text_predictions.jsonl
```

```bash
cellexlink predict-bioc \
  --input examples/sample_input.xml \
  --output outputs/sample_bioc_normalized.xml \
  --ner-output outputs/sample_bioc_ner.xml
```

---

## 8. Optional Docker installation

Build the Docker image from the repository root:

```bash
docker build -t cellexlink -f docker/Dockerfile .
```

Check the CLI:

```bash
docker run --rm cellexlink cellexlink --help
```

Run a small example with a mounted output directory:

```bash
mkdir -p outputs

docker run --rm \
  -v "$(pwd)/examples:/workspace/examples" \
  -v "$(pwd)/outputs:/workspace/outputs" \
  cellexlink \
  cellexlink predict-text \
    --input /workspace/examples/sample_input.txt \
    --output /workspace/outputs/predictions.jsonl
```

---

## 9. Installing benchmark-only baseline tools

Do not install baseline tools such as ScispaCy, BERN2, or VANER2 into the default CellExLink environment unless you specifically need to regenerate comparison results.

Recommended practice:

```text
cellexlink environment      → run CellExLink and package tests
scispacy-baseline env       → run ScispaCy baseline only
bern2-baseline setup        → run BERN2 baseline only
vaner2-baseline setup       → run VANER2 baseline only
```

See `docs/benchmarking.md` and `benchmarks/baselines/README.md` for details.

---

## 10. Common installation checks

Check Python version:

```bash
python --version
```

Check package location:

```bash
python - <<'PY'
import cellexlink
print(cellexlink.__file__)
PY
```

Check PyTorch:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
PY
```

Check Hugging Face access:

```bash
python - <<'PY'
from huggingface_hub import model_info
print(model_info("almire/CellExLink-bioformer16L").modelId)
print(model_info("almire/CellExLink-Sapbert").modelId)
PY
```

---

## 11. Development installation checklist

Before submitting to SoftwareX, confirm that the following commands pass from a clean environment:

```bash
pip install -e ".[dev]"
python -m py_compile src/cellexlink/*.py
python -m py_compile src/cellexlink/io/*.py
python -m py_compile src/cellexlink/recognition/*.py
python -m py_compile src/cellexlink/normalization/*.py
pytest -q
cellexlink --help
python examples/quickstart_text.py --help
python examples/quickstart_bioc.py --help
```

