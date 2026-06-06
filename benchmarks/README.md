# CellExLink benchmarks

This folder contains optional reproducibility scripts for the CellExLink paper. The installable software lives in `src/cellexlink/`; benchmark scripts are kept separate so ordinary users do not need to install baseline tools such as ScispaCy, BERN2, or VANER2.

The benchmarks cover three evaluation levels:

1. **NER evaluation**: exact-span and relaxed-span cell-type recognition.
2. **Gold-span normalization evaluation**: Cell Ontology linking when the gold entity spans are already known.
3. **Strict end-to-end evaluation**: a prediction is correct only when both the mention span and the Cell Ontology identifier are correct.

The scripts accept BioC XML files by default. They can also read the CellExLink JSONL passage schema used by `src/cellexlink/io/`.

## Folder layout

```text
benchmarks/
├── README.md
├── benchmark_utils.py
├── run_cellexlink.py
├── evaluate_ner.py
├── evaluate_nen.py
├── evaluate_end_to_end.py
├── runtime_eval.py
├── make_tables.py
├── results/
└── baselines/
```

The `results/reference_*.csv` files are lightweight copies of the numeric results reported in the manuscript draft. New benchmark runs should write new CSV files, for example `table2_ner_results.csv`.

## Prepare models and datasets

```bash
python scripts/download_models.py --output-dir models
python scripts/download_datasets.py --output-dir data
```

Expected model directories:

```text
models/CellExLink-bioformer16L/
models/CellExLink-Sapbert/
```

Example dataset organization:

```text
data/evaluation/CellLink/test.xml
data/evaluation/CRAFT/test.xml
data/evaluation/BioID/test.xml
data/evaluation/AnatEM/test.xml
data/evaluation/JNLPBA/test.xml
```

Use your actual dataset paths if they differ.

## Run CellExLink predictions

```bash
python benchmarks/run_cellexlink.py \
  --input data/evaluation/CellLink/test.xml \
  --input data/evaluation/CRAFT/test.xml \
  --input data/evaluation/BioID/test.xml \
  --output-dir benchmark_outputs/cellexlink \
  --ner-model models/CellExLink-bioformer16L \
  --nen-model models/CellExLink-Sapbert \
  --mode full
```

This writes files such as:

```text
benchmark_outputs/cellexlink/CellLink_test.ner.xml
benchmark_outputs/cellexlink/CellLink_test.normalized.xml
benchmark_outputs/cellexlink/run_manifest.csv
```

For NER-only benchmarking:

```bash
python benchmarks/run_cellexlink.py \
  --input benchmarks/data/evaluation/JNLPBA/test.xml \
  --output-dir benchmarks/benchmark_outputs/cellexlink_ner \
  --ner-model models/CellExLink-bioformer16L \
  --mode ner
```

For normalization-only benchmarking on gold spans:

```bash
python benchmarks/run_cellexlink.py \
  --input benchmarks/data/evaluation/CRAFT/test.xml \
  --output-dir benchmarks/benchmark_outputs/cellexlink_nen \
  --nen-model models/CellExLink-Sapbert \
  --strip-input-id-infons \
  --mode normalize
```

`--strip-input-id-infons` so the runner creates a temporary spans-only copy
before normalization. This prevents gold ontology IDs from leaking into the prediction file.

## Evaluate NER

```bash
python benchmarks/evaluate_ner.py \
  --system CellExLink \
  --gold CellLink=becnmarks/data/evaluation/CellLink/test.xml \
  --pred CellLink=benchmarks/benchmark_outputs/cellexlink/CellLink_test.ner.xml \
  --output-csv benchmarks/benchmark_outputs/table2_ner_results.csv
```

The script reports both exact and relaxed span scores.

## Evaluate gold-span normalization

```bash
python benchmarks/evaluate_nen.py \
  --system CellExLink \
  --gold CRAFT=benchmarks/data/evaluation/CRAFT/test.xml \
  --pred CRAFT=benchmarks/benchmark_outputs/cellexlink_nen/CRAFT_test.normalized.xml \
  --output-csv benchmarks/benchmark_outputs/table5_gold_span_normalization_results.csv
```

## Evaluate strict end-to-end extraction

```bash
python benchmarks/evaluate_end_to_end.py \
  --system CellExLink \
  --gold CellLink=data/evaluation/CellLink/test.xml \
  --pred CellLink=benchmark_outputs/cellexlink/CellLink_test.normalized.xml \
  --output-csv benchmark_outputs/table7_end_to_end_results.csv
```

## Runtime evaluation

```bash
python benchmarks/runtime_eval.py \
  --input data/evaluation/JNLPBA/test.xml \
  --output-dir benchmark_outputs/runtime \
  --ner-model models/CellExLink-bioformer16L \
  --nen-model models/CellExLink-Sapbert \
  --mode full \
  --repeats 3
```

Runtime depends strongly on hardware, batch size, sequence length, CPU/GPU, and whether model loading is included. Always report hardware and software versions with runtime numbers.

## Regenerate formatted paper tables

```bash
python benchmarks/make_tables.py \
  --results-dir benchmarks/results \
  --output-dir benchmark_outputs/tables
```

## Baselines

Baseline tool support is intentionally optional. See `benchmarks/baselines/` for recommended environment separation and output-normalization guidance.

Do not put ScispaCy, BERN2, VANER2, or their heavy dependencies into `pyproject.toml` base dependencies. Keep them in separate environments.
