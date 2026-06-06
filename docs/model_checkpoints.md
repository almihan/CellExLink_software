# Model checkpoints and resources

CellExLink uses separate checkpoints and resources for recognition and normalization.

---

## 1. Default checkpoints

Default model identifiers:

```text
NER model: almire/CellExLink-bioformer16L
NEN model: almire/CellExLink-Sapbert
```

The NER model detects cell-type spans. The NEN model embeds mentions and ontology aliases for Cell Ontology linking.

---

## 2. Download checkpoints

Use the CellExLink CLI:

```bash
cellexlink download-models --output-dir models
```

Expected structure:

```text
models/
├── CellExLink-bioformer16L/
├── CellExLink-Sapbert/
└── models.json
```

Use local paths:

```bash
cellexlink predict-bioc \
  --input examples/sample_input.xml \
  --output outputs/normalized.xml \
  --ner-model models/CellExLink-bioformer16L \
  --nen-model models/CellExLink-Sapbert
```

---

## 3. Use Hugging Face cache instead of local model folders

You may omit local model paths and use the default Hugging Face model IDs:

```bash
cellexlink predict-text \
  --input examples/sample_input.txt \
  --output outputs/predictions.jsonl
```

Hugging Face will download the models to its cache directory.

To control the cache location:

```bash
export HF_HOME=/path/to/hf_cache
export TRANSFORMERS_CACHE=/path/to/hf_cache/transformers
```

---

## 4. Package resources

CellExLink also uses smaller local resource files:

```text
src/cellexlink/resources/
├── abbreviations.tsv
└── cell_ontology_v2025-12-17.jsonl
```

These files should be included as package data in `pyproject.toml`:

```toml
[tool.setuptools.package-data]
cellexlink = ["resources/*.tsv", "resources/*.jsonl"]
```

The resource filenames should include version or date information when possible.

---

## 5. Specify custom resources

Python:

```python
from cellexlink import CellExLinkPipeline

pipeline = CellExLinkPipeline.from_pretrained(
    ner_model="models/CellExLink-bioformer16L",
    nen_model="models/CellExLink-Sapbert",
    ontology_path="resources/custom_cell_ontology.jsonl",
    abbreviations_path="resources/custom_abbreviations.tsv",
)
```

CLI:

```bash
cellexlink predict-bioc \
  --input examples/sample_input.xml \
  --output outputs/normalized.xml \
  --ontology-path resources/custom_cell_ontology.jsonl \
  --abbreviations-path resources/custom_abbreviations.tsv
```

---

## 6. Disable abbreviation handling

Abbreviation handling is useful for biomedical text, but it can be disabled for controlled experiments.

Python:

```python
pipeline = CellExLinkPipeline.from_pretrained(
    disable_abbreviations=True,
)
```

CLI:

```bash
cellexlink predict-bioc \
  --input examples/sample_input.xml \
  --output outputs/normalized.xml \
  --disable-abbreviations
```

This is useful for ablation studies and debugging.

---

## 7. Reproducibility recommendations

For papers and benchmarks, record:

```text
CellExLink package version
Git commit hash
NER model name or local path
NER model revision or checkpoint date
NEN model name or local path
NEN model revision or checkpoint date
Cell Ontology resource filename and date
abbreviation dictionary filename and date
Python version
PyTorch version
CUDA version, if applicable
hardware used for runtime measurements
```

Example manifest:

```json
{
  "software": "CellExLink",
  "version": "0.1.0",
  "git_commit": "<commit-hash>",
  "ner_model": "almire/CellExLink-bioformer16L",
  "nen_model": "almire/CellExLink-Sapbert",
  "cell_ontology_resource": "cell_ontology_v2025-12-17.jsonl",
  "abbreviation_resource": "abbreviations.tsv",
  "python": "3.11",
  "torch": "2.x",
  "hardware": "CPU or GPU description"
}
```

---

## 8. CPU and GPU use

Small examples can run on CPU, although inference may be slower.

For large corpora, GPU inference is recommended. Increase batch size gradually:

```bash
cellexlink predict-bioc \
  --input data/corpus.xml \
  --output outputs/corpus.normalized.xml \
  --batch-size 16
```

If you encounter GPU memory errors, reduce the batch size:

```bash
cellexlink predict-bioc \
  --input data/corpus.xml \
  --output outputs/corpus.normalized.xml \
  --batch-size 4
```

For compatible GPUs, fp16 may reduce memory use:

```bash
cellexlink predict-bioc \
  --input data/corpus.xml \
  --output outputs/corpus.normalized.xml \
  --fp16
```

---

## 9. Model update policy

Recommended versioning:

```text
v0.1.0 package release       → fixed default model checkpoints
v0.1.1 package patch release → same models, bug fixes only
v0.2.0 package release       → possible model/resource update
```

If a model checkpoint is updated, document it in:

```text
CHANGELOG.md
docs/model_checkpoints.md
GitHub release notes
```

For reproducibility, avoid silently replacing model files associated with a published release.

---

## 10. Do not commit large model files

Do not store Hugging Face checkpoint files in the Git repository.

Add common model/cache patterns to `.gitignore`:

```gitignore
models/
*.bin
*.safetensors
*.pt
*.pth
.cache/
outputs/
```

Use download scripts, release notes, or Hugging Face model IDs instead.

---

## 11. Troubleshooting checkpoint loading

If a model cannot be downloaded:

1. Check internet access.
2. Check whether the model ID is correct.
3. Check whether the model is private or gated.
4. Try downloading with `huggingface-cli`.
5. Try passing a local model directory instead of a Hub ID.

Check model access:

```bash
python - <<'PY'
from huggingface_hub import model_info
print(model_info("almire/CellExLink-bioformer16L"))
print(model_info("almire/CellExLink-Sapbert"))
PY
```

If loading from a local path, confirm the directory contains files such as:

```text
config.json
tokenizer.json
tokenizer_config.json
model.safetensors or pytorch_model.bin
```

