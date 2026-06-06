# Troubleshooting

This page lists common problems and suggested fixes.

---

## 1. `ModuleNotFoundError: No module named 'cellexlink'`

Cause: the package is not installed in the active Python environment.

Fix:

```bash
cd CellExLink
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

Check the active Python:

```bash
which python
python --version
python -m pip show cellexlink
```

---

## 2. `cellexlink: command not found`

Cause: the command-line entry point is not installed or the wrong environment is active.

Fix:

```bash
pip install -e .
```

Then check:

```bash
python -m cellexlink.cli --help
cellexlink --help
```

If `python -m cellexlink.cli --help` works but `cellexlink --help` does not, your environment's script directory may not be on `PATH`.

---

## 3. CLI help works, but prediction fails with missing model files

Cause: model checkpoint could not be found or downloaded.

Fix 1: Download models explicitly.

```bash
cellexlink download-models --output-dir models
```

Then run with local paths:

```bash
cellexlink predict-text \
  --input examples/sample_input.txt \
  --output outputs/predictions.jsonl \
  --ner-model models/CellExLink-bioformer16L \
  --nen-model models/CellExLink-Sapbert
```

Fix 2: Check Hugging Face access.

```bash
python - <<'PY'
from huggingface_hub import model_info
print(model_info("almire/CellExLink-bioformer16L").modelId)
print(model_info("almire/CellExLink-Sapbert").modelId)
PY
```

---

## 4. CUDA out-of-memory error

Cause: prediction batch size is too large for available GPU memory.

Fix: reduce batch size.

```bash
cellexlink predict-bioc \
  --input data/corpus.xml \
  --output outputs/normalized.xml \
  --batch-size 4
```

Try fp16 if your GPU supports it:

```bash
cellexlink predict-bioc \
  --input data/corpus.xml \
  --output outputs/normalized.xml \
  --batch-size 8 \
  --fp16
```

You can also run on CPU for small examples, although it will be slower.

---

## 5. `pyab3p` installation fails

`pyab3p` is recommended for document-level abbreviation expansion, but CellExLink can run without it.

Fix options:

1. Install build tools for your operating system and retry.
2. Use a supported Python version.
3. Run without Ab3P-based long-form recovery.
4. Disable abbreviation handling for debugging.

Disable abbreviation handling:

```bash
cellexlink predict-bioc \
  --input examples/sample_input.xml \
  --output outputs/normalized.xml \
  --disable-abbreviations
```

---

## 6. BioC XML parsing error

Cause: malformed XML, invalid characters, or missing closing tags.

Check XML validity:

```bash
python - <<'PY'
from xml.etree import ElementTree as ET
ET.parse("examples/sample_input.xml")
print("XML parsed successfully")
PY
```

Common issues:

```text
unescaped ampersand: use &amp;
missing closing tag
invalid encoding
empty text node
```

---

## 7. No predictions are returned

Possible causes:

1. The input text does not contain recognizable cell-type mentions.
2. The NER model did not load correctly.
3. The input BioC passages are empty.
4. Offsets or text fields are missing from BioC XML.
5. The wrong input file was provided.

Debug steps:

```bash
cellexlink predict-bioc \
  --input examples/sample_input.xml \
  --output outputs/normalized.xml \
  --ner-output outputs/ner_predictions.xml
```

Then inspect:

```bash
head -n 50 outputs/ner_predictions.xml
head -n 50 outputs/normalized.xml
```

Check that passages are readable:

```bash
python - <<'PY'
from cellexlink.io import iter_bioc_passages
for p in iter_bioc_passages("examples/sample_input.xml"):
    print(p.document_id, p.passage_id, repr(p.text[:80]))
PY
```

---

## 8. Normalization output has mentions but no CL IDs

Possible causes:

1. The normalization model did not load.
2. The Cell Ontology resource file is missing.
3. Annotation text is empty.
4. The annotation type was not recognized by the normalization code.
5. The input XML does not contain NER annotations.

Debug with normalization-only mode:

```bash
cellexlink normalize-bioc \
  --input outputs/ner_predictions.xml \
  --output outputs/normalized_debug.xml \
  --nen-model almire/CellExLink-Sapbert
```

Specify resources explicitly:

```bash
cellexlink normalize-bioc \
  --input outputs/ner_predictions.xml \
  --output outputs/normalized_debug.xml \
  --ontology-path src/cellexlink/resources/cell_ontology_v2025-12-17.jsonl \
  --abbreviations-path src/cellexlink/resources/abbreviations.tsv
```

---

## 9. Offset mismatch in output BioC XML

Cause: BioC offsets are absolute document offsets, while many internal functions work with passage-relative offsets.

Check passage offset:

```xml
<passage>
  <offset>100</offset>
  <text>CD8+ T cells were detected.</text>
</passage>
```

A mention beginning at character `0` in the passage should have absolute BioC offset `100`.

If offsets are wrong, confirm that your input BioC XML has correct `<offset>` values for every passage.

---

## 10. Import works in Python, but tests fail

Possible causes:

1. Old files from the previous structure are still being imported.
2. The package was not reinstalled after moving files.
3. There are duplicate package folders.

Fix:

```bash
pip uninstall cellexlink -y
pip install -e ".[dev]"
pytest -q
```

Check import path:

```bash
python - <<'PY'
import cellexlink
print(cellexlink.__file__)
PY
```

It should point to your current repository.

---

## 11. Training is very slow

Training transformer models can be expensive.

Suggestions:

```text
use a GPU
reduce max sequence length
reduce batch size
increase gradient accumulation instead of batch size
use a smaller validation subset while debugging
save checkpoints less frequently
run smoke tests before full training
```

For debugging, run one epoch or fewer steps first:

```bash
python -m cellexlink.recognition.train \
  --model-path bioformers/bioformer-16L \
  --train-xml data/train.xml \
  --validation-xml data/validation.xml \
  --output-dir models/debug-ner \
  --num-train-epochs 1 \
  --max-train-samples 100 \
  --max-eval-samples 50
```

Only use full training once the debug run succeeds.

---

## 12. Baseline dependency conflicts

Do not install ScispaCy, BERN2, VANER2, and CellExLink into the same environment unless you have confirmed that the dependency versions are compatible.

Recommended approach:

```text
cellexlink-env        → CellExLink package and tests
scispacy-env          → ScispaCy baseline
bern2-env             → BERN2 baseline
vaner2-env            → VANER2 baseline
```

Keep baseline dependencies under:

```text
benchmarks/baselines/
```

not in the core package requirements.

---

## 13. Clean generated files

To remove generated outputs and caches:

```bash
rm -rf outputs/
rm -rf cellexlink_outputs/
rm -rf .pytest_cache/
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
```

Do not delete downloaded model directories unless you want to re-download the checkpoints.

---

## 14. What to include in bug reports

When reporting a problem, include:

```text
CellExLink version
Git commit hash
Python version
operating system
installation command
exact command that failed
full error traceback
small input example, if possible
whether you used CPU or GPU
model paths or model IDs
```

Useful diagnostic command:

```bash
python - <<'PY'
import platform
import cellexlink
print("CellExLink:", cellexlink.__version__)
print("Python:", platform.python_version())
print("Platform:", platform.platform())
try:
    import torch
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
except Exception as exc:
    print("Torch unavailable:", exc)
PY
```

