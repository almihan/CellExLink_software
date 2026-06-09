# VANER2 baseline

Wrapper for running the external VANER2 NER model on the CellExLink benchmark datasets.

Official VANER2 repository: <https://github.com/ZhuLab-Fudan/VANER2>

## Setup

Install VANER2 in a separate environment following the official repository instructions. Copy the required VANER2 files into this directory:

```text
benchmarks/baselines/vaner2/
├── run_vaner2.py
├── model/                         # from VANER2
├── finetuned_models/VANER2/saved.pt
├── data/                          # temporary PubTator inputs
├── results/                       # temporary VANER2 outputs
└── model_outputs/                 # final BioC XML predictions
```

VANER2 inference requires a CUDA GPU.

## Run predictions

Run all benchmark datasets:

```bash
cd benchmarks/baselines/vaner2
conda activate vaner2
python run_vaner2.py --model_names VANER2
```

Run one dataset:

```bash
python run_vaner2.py \
  --input_xml ../../data/evaluation/AnatEM/test.xml \
  --model_names VANER2
```

Final XML predictions are written to:

```text
benchmarks/baselines/vaner2/model_outputs/
```

## Evaluate NER results

Run evaluation from the CellExLink repository root.

Single dataset example:

```bash
python benchmarks/evaluate_ner.py \
  --gold AnatEM=benchmarks/data/evaluation/AnatEM/test.xml \
  --pred AnatEM=benchmarks/baselines/vaner2/model_outputs/AnatEM_test.xml \
  --system VANER2 \
  --output-csv benchmarks/baselines/vaner2/model_outputs/vaner2_AnatEM_ner_results.csv
```

All benchmark datasets:

```bash
DATASETS=(AnatEM BioID CRAFT Celllink JNLPBA)
ARGS=()
for d in "${DATASETS[@]}"; do
  ARGS+=(--gold "$d=benchmarks/data/evaluation/$d/test.xml")
  ARGS+=(--pred "$d=benchmarks/baselines/vaner2/model_outputs/${d}_test.xml")
done

python benchmarks/evaluate_ner.py "${ARGS[@]}" \
  --system VANER2 \
  --macro-average \
  --output-csv benchmarks/baselines/vaner2/model_outputs/vaner2_ner_results.csv
```

## Notes

- VANER2 is used here as an NER-only baseline.
- Keep the final prediction XML files in `model_outputs/` for reproducibility.
- Use `evaluate_ner.py`; do not use NEN or end-to-end evaluators for this baseline.
