# Docker usage for CellExLink

This Docker setup provides a reproducible Linux environment for CellExLink.
It does not copy model checkpoints, datasets, or benchmark outputs into the
image. Mount those directories at runtime.

## Build the image

```bash
docker build -t cellexlink:0.1.0 .
```

For development/testing inside the container:

```bash
docker build --build-arg INSTALL_EXTRAS=dev -t cellexlink:dev .
docker run --rm cellexlink:dev pytest -q /opt/cellexlink/tests
```

## Check the command-line interface

```bash
docker run --rm cellexlink:0.1.0 cellexlink --help
```

## Run end-to-end BioC extraction

Assuming local directories:

```text
models/
  CellExLink-bioformer16L/
  CellExLink-Sapbert/
examples/
  sample_input.xml
outputs/
```

Run:

```bash
mkdir -p outputs

docker run --rm \
  -v "$PWD/models:/models:ro" \
  -v "$PWD/examples:/examples:ro" \
  -v "$PWD/outputs:/outputs" \
  cellexlink:0.1.0 \
  cellexlink predict-bioc \
    --input /examples/sample_input.xml \
    --output /outputs/sample.end_to_end.xml \
    --ner-output /outputs/sample.ner.xml \
    --ner-model /models/CellExLink-bioformer16L \
    --nen-model /models/CellExLink-Sapbert
```

## Run gold-span NEN only

Use this when the input BioC XML already contains mention annotations and you
only want Cell Ontology normalization.

```bash
docker run --rm \
  -v "$PWD/models:/models:ro" \
  -v "$PWD/examples:/examples:ro" \
  -v "$PWD/outputs:/outputs" \
  cellexlink:0.1.0 \
  cellexlink normalize-bioc \
    --input /examples/sample_gold_spans.xml \
    --output /outputs/sample.normalized.xml \
    --nen-model /models/CellExLink-Sapbert
```

## Run benchmark prediction files

```bash
mkdir -p benchmark_outputs

docker run --rm \
  -v "$PWD/models:/models:ro" \
  -v "$PWD/data:/data:ro" \
  -v "$PWD/benchmark_outputs:/benchmark_outputs" \
  cellexlink:0.1.0 \
  python /opt/cellexlink/benchmarks/run_cellexlink.py \
    --mode full \
    --input /data/evaluation/BioID/test.xml \
    --output-dir /benchmark_outputs/cellexlink/end_to_end \
    --ner-model /models/CellExLink-bioformer16L \
    --nen-model /models/CellExLink-Sapbert
```

## Use docker compose

```bash
mkdir -p models data outputs
docker compose build
docker compose run --rm cellexlink cellexlink --help
```

Edit `docker-compose.yml` to change the command for your input files.

## GPU note

The default Dockerfile is CPU-compatible. For NVIDIA GPU execution, install the
NVIDIA Container Toolkit on the host and run the same image with:

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/models:ro" \
  -v "$PWD/examples:/examples:ro" \
  -v "$PWD/outputs:/outputs" \
  cellexlink:0.1.0 \
  cellexlink predict-bioc \
    --input /examples/sample_input.xml \
    --output /outputs/sample.end_to_end.xml \
    --ner-output /outputs/sample.ner.xml \
    --ner-model /models/CellExLink-bioformer16L \
    --nen-model /models/CellExLink-Sapbert
```

For a production GPU image, you may later switch the base image to a PyTorch
CUDA runtime image, but keep this CPU image as the minimal reproducible default.
