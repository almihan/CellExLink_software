# ScispaCy baseline for CellExLink benchmarks

This folder contains optional scripts for running the ScispaCy baseline used in
CellExLink benchmark comparisons. These scripts are **not part of the
installable `cellexlink` package**. They are provided only as reproducibility
material for the SoftwareX paper and benchmark folder.

## What is included

- `predict_ner.py`: runs a ScispaCy NER model on BioC XML and writes predicted
  cell-type spans to BioC XML.
- `predict_nen.py`: links existing BioC annotations to Cell Ontology identifiers
  using the ScispaCy/PyOBO linker.
- `run_scispacy.py`: convenience wrapper that runs NER-only, normalization-only,
  or full NER+NEN baseline workflows.
- `environment.yml`: optional Conda environment for ScispaCy benchmarking.

## Why this is separate

ScispaCy, spaCy models, and PyOBO can have dependency constraints that are
different from CellExLink's PyTorch/Transformers runtime. Keep this baseline in
a separate environment.

## Install baseline environment

From the repository root:

```bash
conda env create -f benchmarks/baselines/scispacy/environment.yml
conda activate cellexlink-scispacy
```

Then install the ScispaCy model used for cell ontology recognition:

```bash
python -m pip install scispacy
python -m pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.6.0/en_ner_craft_md-0.6.0.tar.gz
```

If the model URL changes, use the current model download link from the ScispaCy
model page.

## Recommended benchmark data convention

Use text-only input files for prediction and gold files only for evaluation:

```text
benchmarks/data/evaluation/CellLink/input.xml
benchmarks/data/evaluation/CellLink/gold.xml
benchmarks/data/evaluation/CellLink/gold_spans.xml
```

## Run ScispaCy NER-only baseline

```bash
python benchmarks/baselines/scispacy/run_scispacy.py \
  --input benchmarks/data/evaluation/JNLPBA/test.xml \
  --output-dir benchmark_outputs/scispacy \
  --mode ner
```

This writes:

```text
benchmark_outputs/scispacy/JNLPBA_input.scispacy.ner.xml
```

Evaluate it with the shared CellExLink benchmark evaluator:

```bash
python benchmarks/evaluate_ner.py \
  --system ScispaCy \
  --gold JNLPBA=benchmarks/data/evaluation/JNLPBA/test.xml \
  --pred JNLPBA=benchmark_outputs/scispacy/JNLPBA_test.scispacy.ner.xml \
  --output-csv benchmark_outputs/scispacy_ner_results.csv
```

## Run ScispaCy normalization-only baseline

Use `gold_spans.xml` as input. This evaluates linking separately from NER.

```bash
python benchmarks/baselines/scispacy/run_scispacy.py \
  --input benchmarks/data/evaluation/CellLink/gold_spans.xml \
  --output-dir benchmark_outputs/scispacy_nen \
  --mode normalize
```

This writes:

```text
benchmark_outputs/scispacy_nen/CellLink_gold_spans.scispacy.normalized.xml
```

Evaluate it:

```bash
python benchmarks/evaluate_nen.py \
  --system ScispaCy \
  --gold CellLink=benchmarks/data/evaluation/CellLink/gold.xml \
  --pred CellLink=benchmark_outputs/scispacy_nen/CellLink_gold_spans.scispacy.normalized.xml \
  --output-csv benchmark_outputs/scispacy_nen_results.csv
```

## Run full ScispaCy NER + NEN baseline

```bash
python benchmarks/baselines/scispacy/run_scispacy.py \
  --input benchmarks/data/evaluation/CellLink/input.xml \
  --input benchmarks/data/evaluation/CRAFT/input.xml \
  --input benchmarks/data/evaluation/BioID/input.xml \
  --output-dir benchmark_outputs/scispacy \
  --mode full
```

This writes both NER and normalized outputs:

```text
benchmark_outputs/scispacy/CellLink_input.scispacy.ner.xml
benchmark_outputs/scispacy/CellLink_input.scispacy.normalized.xml
```

Evaluate strict end-to-end performance:

```bash
python benchmarks/evaluate_end_to_end.py \
  --system ScispaCy \
  --gold CellLink=benchmarks/data/evaluation/CellLink/gold.xml \
  --pred CellLink=benchmark_outputs/scispacy/CellLink_input.scispacy.normalized.xml \
  --gold CRAFT=benchmarks/data/evaluation/CRAFT/gold.xml \
  --pred CRAFT=benchmark_outputs/scispacy/CRAFT_input.scispacy.normalized.xml \
  --gold BioID=benchmarks/data/evaluation/BioID/gold.xml \
  --pred BioID=benchmark_outputs/scispacy/BioID_input.scispacy.normalized.xml \
  --output-csv benchmark_outputs/scispacy_end_to_end_results.csv
```

