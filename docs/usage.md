# Usage

This guide shows how to use CellExLink from Python and from the command line.

CellExLink supports three common workflows:

1. End-to-end extraction from plain text.
2. End-to-end extraction from BioC XML.
3. Separate recognition-only or normalization-only execution.

---

## 1. Python quick start: plain text

```python
from cellexlink import CellExLinkPipeline

pipeline = CellExLinkPipeline.from_pretrained()

results = pipeline.extract_text(
    "The mesothelial cell and SMC clusters formed the third population."
)

for item in results:
    print(item.to_dict())
```

Each result is an `ExtractionResult` with fields such as:

```text
document_id
passage_index
mention
start
end
entity_type
cl_id
cl_label
score
source
```

---

## 2. Python quick start: text file

```python
from cellexlink import CellExLinkPipeline

pipeline = CellExLinkPipeline.from_pretrained()

pipeline.extract_text_file(
    input_txt="examples/sample_input.txt",
    output_jsonl="outputs/text_predictions.jsonl",
)
```

The output is a JSONL file with one linked mention per line.

---

## 3. Python quick start: BioC XML

```python
from cellexlink import CellExLinkPipeline

pipeline = CellExLinkPipeline.from_pretrained()

pipeline.extract_bioc(
    input_xml="examples/sample_input.xml",
    output_xml="outputs/normalized.xml",
    ner_output_xml="outputs/ner_predictions.xml",
)
```

This runs the complete pipeline:

```text
BioC input XML
  ↓
cell-type recognition
  ↓
NER BioC XML
  ↓
Cell Ontology normalization
  ↓
normalized BioC XML
```

---

## 4. Use local checkpoints

After downloading models:

```bash
cellexlink download-models --output-dir models
```

Use local model paths:

```python
from cellexlink import CellExLinkPipeline

pipeline = CellExLinkPipeline.from_pretrained(
    ner_model="models/CellExLink-bioformer16L",
    nen_model="models/CellExLink-Sapbert",
)

pipeline.extract_bioc(
    input_xml="examples/sample_input.xml",
    output_xml="outputs/normalized.xml",
)
```

---

## 5. Command line: plain text string

```bash
cellexlink predict-text \
  --text "CD8+ T cells were enriched in the tumor microenvironment." \
  --output outputs/predictions.jsonl
```

Print the result:

```bash
cat outputs/predictions.jsonl
```

---

## 6. Command line: plain text file

```bash
cellexlink predict-text \
  --input examples/sample_input.txt \
  --output outputs/text_predictions.jsonl
```

With local models:

```bash
cellexlink predict-text \
  --input examples/sample_input.txt \
  --output outputs/text_predictions.jsonl \
  --ner-model models/CellExLink-bioformer16L \
  --nen-model models/CellExLink-Sapbert
```

---

## 7. Command line: BioC XML

```bash
cellexlink predict-bioc \
  --input examples/sample_input.xml \
  --output outputs/normalized.xml \
  --ner-output outputs/ner_predictions.xml
```

With local models:

```bash
cellexlink predict-bioc \
  --input examples/sample_input.xml \
  --output outputs/normalized.xml \
  --ner-output outputs/ner_predictions.xml \
  --ner-model models/CellExLink-bioformer16L \
  --nen-model models/CellExLink-Sapbert
```

---

## 8. Recognition only

Use recognition-only mode when you want detected cell-type spans but do not need Cell Ontology identifiers yet.

Python:

```python
from cellexlink.recognition import predict_ner

predict_ner(
    model_path="almire/CellExLink-bioformer16L",
    input_xml="examples/sample_input.xml",
    output_dir="outputs/ner_work",
    output_xml="outputs/ner_predictions.xml",
)
```

Module command:

```bash
python -m cellexlink.recognition.predict \
  --model-path almire/CellExLink-bioformer16L \
  --input-xml examples/sample_input.xml \
  --output-dir outputs/ner_work \
  --output-xml outputs/ner_predictions.xml
```

---

## 9. Normalization only

Use normalization-only mode when your BioC XML already contains cell-type annotations.

Python:

```python
from cellexlink.normalization import normalize_bioc

normalize_bioc(
    input_xml="outputs/ner_predictions.xml",
    output_xml="outputs/normalized.xml",
    model_path="almire/CellExLink-Sapbert",
)
```

CLI:

```bash
cellexlink normalize-bioc \
  --input outputs/ner_predictions.xml \
  --output outputs/normalized.xml \
  --nen-model almire/CellExLink-Sapbert
```

Module command:

```bash
python -m cellexlink.normalization.linker \
  outputs/ner_predictions.xml \
  outputs/normalized.xml \
  --model-path almire/CellExLink-Sapbert
```

---

## 10. Work with package I/O utilities

Create a BioC XML file from plain text:

```python
from cellexlink.io import write_text_as_bioc

write_text_as_bioc(
    text="Trophoblast progenitor cells showed altered signaling.",
    output_xml="outputs/input.xml",
    document_id="example-doc",
)
```

Convert BioC XML to passage JSONL:

```python
from cellexlink.io import convert_bioc_to_jsonl

convert_bioc_to_jsonl(
    srcs="examples/sample_input.xml",
    output_jsonl="outputs/passages.jsonl",
    include_entities=False,
)
```

Read passages:

```python
from cellexlink.io import iter_bioc_passages

for passage in iter_bioc_passages("examples/sample_input.xml"):
    print(passage.document_id, passage.passage_id, passage.text)
```

---

## 11. Fine-tune the NER model

Example:

```bash
python -m cellexlink.recognition.train \
  --model-path bioformers/bioformer-16L \
  --train-xml data/train.xml \
  --validation-xml data/validation.xml \
  --output-dir models/cellexlink-ner \
  --num-train-epochs 3 \
  --per-device-train-batch-size 8 \
  --overwrite-output-dir
```

Use JSONL instead of BioC XML:

```bash
python -m cellexlink.recognition.train \
  --model-path bioformers/bioformer-16L \
  --train-file data/train.jsonl \
  --validation-file data/validation.jsonl \
  --output-dir models/cellexlink-ner
```

---

## 12. Fine-tune the NEN model

Generate training pairs from the ontology resource:

```bash
python -m cellexlink.normalization.train create-pairs \
  --cell-types src/cellexlink/resources/cell_ontology_v2025-12-17.jsonl \
  --output data/nen_pairs.tsv
```

Train a SapBERT-style sentence-transformers model:

```bash
python -m cellexlink.normalization.train train \
  --model-name cambridgeltl/SapBERT-from-PubMedBERT-fulltext \
  --train-pairs data/nen_pairs.tsv \
  --output-dir models/cellexlink-sapbert \
  --epochs 3
```

Training is optional for normal users. Most users should use the released CellExLink checkpoints.

---

## 13. Output locations

Typical output files:

```text
outputs/
├── text_predictions.jsonl
├── ner_predictions.xml
├── normalized.xml
└── ner_work/
    ├── predictions.json
    ├── predictions.jsonl
    ├── predict_results.json
    └── predict_runtime_summary.json
```

For large-scale processing, create a separate output directory for each corpus or batch.

---

## 14. Recommended usage for large corpora

Use BioC XML input when processing biomedical articles, abstracts, captions, or curated corpora.

Recommended pattern:

```bash
mkdir -p outputs/my_corpus

cellexlink predict-bioc \
  --input data/my_corpus.xml \
  --output outputs/my_corpus/normalized.xml \
  --ner-output outputs/my_corpus/ner_predictions.xml \
  --output-dir outputs/my_corpus/work \
  --batch-size 32
```

For GPU inference, increase `--batch-size` gradually until memory usage is stable.

For CPU inference, use a smaller batch size:

```bash
cellexlink predict-bioc \
  --input data/my_corpus.xml \
  --output outputs/my_corpus/normalized.xml \
  --batch-size 4
```

---

## 15. Recommended usage in downstream pipelines

The simplest downstream integration is to read the final normalized BioC XML and extract each annotation with its Cell Ontology ID.

If you prefer JSONL, use the Python API:

```python
from cellexlink import CellExLinkPipeline
from cellexlink.pipeline import write_predictions_jsonl

pipeline = CellExLinkPipeline.from_pretrained()

pipeline.extract_bioc(
    input_xml="examples/sample_input.xml",
    output_xml="outputs/normalized.xml",
)

predictions = pipeline.read_predictions_from_bioc("outputs/normalized.xml")
write_predictions_jsonl(predictions, "outputs/normalized_predictions.jsonl")
```

