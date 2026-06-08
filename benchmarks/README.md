# Minimal benchmark evaluation for the SoftwareX CellExLink release

Use the original-compatible evaluator for the manuscript numbers.  The package
can keep other helper scripts for debugging, but the reported NEN and strict
end-to-end tables should be produced by these two commands.

## run end-to-end
```bash
python benchmarks/run_cellexlink.py \
  --mode full \
  --input benchmarks/data/evaluation/Celllink/test.xml \
  --output-dir benchmarks/benchmark_outputs/cellexlink/end_to_end \
  --ner-model models/CellExLink-bioformer16L \
  --nen-model models/CellExLink-Sapbert \
  --batch-size 16 \
  --manifest benchmarks/benchmark_outputs/cellexlink/end_to_end/run_manifest.csv
```

## run NER 
```bash
python benchmarks/run_cellexlink.py \
  --mode ner \
  --input benchmarks/data/evaluation/CRAFT/test.xml \
  --output-dir benchmarks/benchmark_outputs/cellexlink/ner/
```

## run NEN 
```bash
python benchmarks/run_cellexlink.py \
  --mode normalize \
  --input benchmarks/data/evaluation/CRAFT/test.xml \
  --output-dir benchmarks/benchmark_outputs/cellexlink/nen
```

## NER evaluation
```bash
python benchmarks/evaluate_ner.py \
  --gold CRAFT=benchmarks/data/evaluation/CRAFT/test.xml \
  --pred CRAFT=benchmarks/benchmark_outputs/cellexlink/ner/CRAFT_test.ner.xml \
  --system CellExLink \
  --output-csv benchmarks/benchmark_outputs/table_ner_results.csv
```




## Gold-span NEN

Do **not** strip the gold `identifier` infons from the normalization input when
you want to run the original-compatible gold-span NEN evaluator.  The original
CellExLink NEN evaluator iterates over gold annotations in the prediction XML
and reads predicted fields such as `CellExLink-Sapbert_id_0`.
--dataset-style other, for dataset otherthan cellink.

```bash
python benchmarks/evaluate_nen.py \
  --dataset-style celllink \
  --gold Celllink=benchmarks/data/evaluation/Celllink/test.xml \
  --pred Celllink=benchmarks/benchmark_outputs/cellexlink/nen/Celllink_test.normalized.xml \
  --model-names CellExLink-Sapbert \
  --output-csv benchmarks/benchmark_outputs/table5_gold_span_normalization_results.csv
```

For CellLink, use `--dataset-style celllink`.

## Strict end-to-end

```bash
python benchmarks/evaluate_end_to_end.py \
  --dataset-style celllink \
  --gold BioID=benchmarks/data/evaluation/BioID/test.xml \
  --pred BioID=benchmarks/benchmark_outputs/cellexlink/end_to_end/BioID_test.normalized.xml \
  --model-names CellExLink-Sapbert \
  --output-csv benchmarks/benchmark_outputs/table7_end_to_end_results.csv
```

For CellLink, use `--dataset-style celllink`.
