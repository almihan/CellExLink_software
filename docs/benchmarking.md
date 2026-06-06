# Benchmarking

This guide describes how to organize CellExLink benchmarking and how to keep comparison code separate from the core package.

The core package should remain easy to install. External baseline tools should be optional because they can require different dependencies, different Python versions, command-line servers, or GPU configurations.

---

## 1. Benchmarking goals

CellExLink can be evaluated at three levels:

1. **Cell-type recognition**: predicted span is compared with the gold span.
2. **Cell Ontology normalization on gold spans**: the gold mention span is given, and the system must predict the correct CL ID.
3. **Strict end-to-end extraction**: both the mention span and the CL ID must be correct.

These levels correspond to the two-stage CellExLink pipeline: NER first, then NEN.

---

## 2. Recommended benchmark folder structure

```text
benchmarks/
├── README.md
├── run_cellexlink.py
├── evaluate_ner.py
├── evaluate_nen.py
├── evaluate_end_to_end.py
├── runtime_eval.py
├── make_tables.py
├── data/
│   └── README.md
├── predictions/
│   ├── cellexlink/
│   ├── scispacy/
│   ├── bern2/
│   └── vaner2/
├── results/
│   ├── table_ner_results.csv
│   ├── table_nen_results.csv
│   ├── table_end_to_end_results.csv
│   └── runtime_results.csv
└── baselines/
    ├── README.md
    ├── scispacy/
    ├── bern2/
    └── vaner2/
```

Do not put baseline code inside `src/cellexlink/`.

---

## 3. Benchmark levels

### Level 1: CellExLink-only evaluation

This level evaluates only CellExLink. It should run in the normal CellExLink environment.

```bash
python benchmarks/run_cellexlink.py \
  --input data/celllink_test.xml \
  --output-dir benchmarks/predictions/cellexlink/celllink
```

```bash
python benchmarks/evaluate_ner.py \
  --gold data/celllink_test.xml \
  --pred benchmarks/predictions/cellexlink/celllink/ner_predictions.xml \
  --output benchmarks/results/celllink_ner.csv
```

```bash
python benchmarks/evaluate_end_to_end.py \
  --gold data/celllink_test.xml \
  --pred benchmarks/predictions/cellexlink/celllink/normalized.xml \
  --output benchmarks/results/celllink_end_to_end.csv
```

### Level 2: Recreate manuscript tables from stored predictions

This level uses stored prediction files and does not require rerunning every baseline.

```bash
python benchmarks/make_tables.py \
  --results-dir benchmarks/results \
  --output-dir benchmarks/tables
```

### Level 3: Regenerate all external baselines

This level requires separate setup for ScispaCy, BERN2, VANER2, or other tools.

```text
benchmarks/baselines/scispacy/
benchmarks/baselines/bern2/
benchmarks/baselines/vaner2/
```

Each baseline should have its own README and environment file.

---

## 4. Recognition metrics

Recommended recognition metrics:

```text
precision
recall
micro-F1
macro average across corpora
```

Span matching modes:

| Mode | Definition |
|---|---|
| Exact span | Prediction is correct only if start and end offsets exactly match the gold span. |
| Relaxed span | Prediction is correct if the predicted span overlaps the gold span under the selected token/character overlap rule. |

Recommended output columns:

```text
dataset,system,matching,precision,recall,f1,tp,fp,fn
```

Example:

```csv
dataset,system,matching,precision,recall,f1,tp,fp,fn
CellLink,CellExLink,exact,0.86,0.86,0.86,100,15,18
```

---

## 5. Normalization metrics on gold spans

For gold-span normalization, the mention boundaries are fixed by the gold data. The system must assign the correct Cell Ontology ID.

Recommended output columns:

```text
dataset,system,condition,accuracy,precision,recall,f1,correct,total
```

Useful conditions:

```text
exact-match-only
all-ids
cell phenotype
heterogeneous cell population
overall
```

---

## 6. Strict end-to-end metrics

A prediction is correct only when both conditions hold:

```text
predicted span matches the gold span
predicted CL ID matches the gold CL ID
```

Recommended output columns:

```text
dataset,system,precision,recall,f1,tp,fp,fn
```

This is the most important practical metric for downstream literature-mining workflows.

---

## 7. Runtime metrics

Recommended runtime reporting:

```text
model name
model size
hardware
batch size
number of documents
number of passages
number of mentions
runtime seconds
milliseconds per abstract or passage
milliseconds per entity
```

Recommended output columns:

```text
task,system,model,hardware,batch_size,n_items,total_seconds,ms_per_item
```

Tasks:

```text
NER per abstract or passage
NEN per entity
end-to-end per document
```

Record whether the runtime used CPU or GPU.

---

## 8. Dataset organization

Do not commit full benchmark datasets to the repository unless redistribution is clearly allowed.

Recommended structure:

```text
data/
├── README.md
├── raw/
├── processed/
└── checksums.txt
```

Add a script:

```bash
python scripts/download_datasets.py --output data/raw
```

Document:

```text
dataset name
source URL or DOI
license
original citation
expected file names
preprocessing command
checksum, if possible
```

---

## 9. Baseline tools

Keep baseline tools outside the core package.

Recommended baseline subfolders:

```text
benchmarks/baselines/scispacy/
benchmarks/baselines/bern2/
benchmarks/baselines/vaner2/
```

Each baseline folder should contain:

```text
README.md
environment.yml or requirements.txt
run_baseline.py or conversion scripts
expected input format
expected output format
```

Do not add baseline dependencies to the default `pyproject.toml` dependencies.

---

## 10. Stored baseline predictions

For reproducibility, you may store small processed prediction files if licensing allows.

Recommended format:

```text
benchmarks/predictions/<system>/<dataset>/predictions.xml
benchmarks/predictions/<system>/<dataset>/predictions.jsonl
benchmarks/predictions/<system>/<dataset>/metadata.json
```

Each `metadata.json` should record:

```json
{
  "system": "CellExLink",
  "version": "0.1.0",
  "model": "almire/CellExLink-bioformer16L + almire/CellExLink-Sapbert",
  "dataset": "CellLink",
  "date_run": "YYYY-MM-DD",
  "command": "...",
  "hardware": "..."
}
```

---

## 11. Example complete CellExLink benchmark run

```bash
mkdir -p benchmarks/predictions/cellexlink/celllink
mkdir -p benchmarks/results

python benchmarks/run_cellexlink.py \
  --input data/processed/celllink_test.xml \
  --output-dir benchmarks/predictions/cellexlink/celllink \
  --ner-model models/CellExLink-bioformer16L \
  --nen-model models/CellExLink-Sapbert

python benchmarks/evaluate_ner.py \
  --gold data/processed/celllink_test.xml \
  --pred benchmarks/predictions/cellexlink/celllink/ner_predictions.xml \
  --matching exact \
  --output benchmarks/results/celllink_ner_exact.csv

python benchmarks/evaluate_end_to_end.py \
  --gold data/processed/celllink_test.xml \
  --pred benchmarks/predictions/cellexlink/celllink/normalized.xml \
  --output benchmarks/results/celllink_end_to_end.csv
```

---

## 12. Recommended SoftwareX presentation

For the manuscript, keep benchmarking concise:

```text
one table for recognition
one table for gold-span normalization
one table for strict end-to-end extraction
one small runtime table or figure
```

Move detailed baseline setup, large intermediate outputs, and ablation scripts to `benchmarks/` or supplementary material.

The submitted software should remain easy to install and test without installing every external baseline system.

