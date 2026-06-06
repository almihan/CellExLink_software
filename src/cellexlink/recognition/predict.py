from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
from transformers import TrainingArguments
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from .bioc import convert_bioc_to_json, write_predictions_to_bioc_xml
from .hf_utils import (
    HFModelOptions,
    load_fast_tokenizer,
    load_model_config,
    load_token_classification_model,
    resolve_effective_max_seq_length,
    resolve_model_reference,
    synchronize_if_cuda,
)
from .ner_common import (
    GENERATED_JSON_FILENAMES,
    DatasetColumns,
    TokenClassificationBatchCollator,
    build_label_schema_from_model,
    build_trainer,
    configure_logging,
    infer_columns,
    load_raw_datasets,
    preprocess_prediction_dataset,
    reconstruct_prediction_outputs,
    save_prediction_outputs,
)

check_min_version("4.37.0")
require_version("datasets>=1.8.0", "Please install a compatible `datasets` version for CellExLink.")

LOGGER = logging.getLogger(__name__)
PathLike = str | Path
RUNTIME_SUMMARY_FILENAME = "predict_runtime_summary.json"


def _build_prediction_training_args(
    *,
    output_dir: PathLike,
    per_device_predict_batch_size: int,
    fp16: bool,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        do_predict=True,
        per_device_eval_batch_size=per_device_predict_batch_size,
        remove_unused_columns=False,
        report_to=[],
        fp16=fp16,
    )


def predict_ner(
    *,
    model_path: PathLike,
    input_xml: Optional[PathLike] = None,
    input_file: Optional[PathLike] = None,
    output_dir: PathLike,
    output_xml: Optional[PathLike] = None,
    tokenizer_path: Optional[PathLike] = None,
    text_column_name: Optional[str] = None,
    max_seq_length: Optional[int] = None,
    doc_stride: int = 128,
    pad_to_max_length: bool = False,
    max_predict_samples: Optional[int] = None,
    warmup_runs: int = 1,
    per_device_predict_batch_size: int = 16,
    preprocessing_num_workers: Optional[int] = None,
    overwrite_cache: bool = False,
    cache_dir: Optional[PathLike] = None,
    model_revision: str = "main",
    token: Optional[str] = None,
    trust_remote_code: bool = False,
    fp16: bool = False,
) -> int:
    """
    Run CellExLink cell-type NER prediction.

    Input can be either:
    1. BioC XML via `input_xml`; or
    2. JSON/JSONL/CSV via `input_file` with at least a text column.

    Outputs written to `output_dir`:
    - predictions.json
    - predictions.jsonl
    - predict_runtime_summary.json

    If `output_xml` is provided with `input_xml`, predicted annotations are also
    inserted back into a BioC XML copy.
    """
    if (input_xml is None) == (input_file is None):
        raise ValueError("Provide exactly one of `input_xml` or `input_file`.")
    if output_xml is not None and input_xml is None:
        raise ValueError("`output_xml` requires `input_xml`, because BioC export needs the original XML structure.")
    if warmup_runs < 0:
        raise ValueError("`warmup_runs` must be >= 0.")
    if doc_stride < 0:
        raise ValueError("`doc_stride` must be >= 0.")

    output_dir_path = Path(output_dir).resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)

    if input_xml is not None:
        input_xml_path = Path(input_xml).resolve()
        if not input_xml_path.is_file():
            raise FileNotFoundError(f"Missing input XML: {input_xml_path}")
        test_file = output_dir_path / GENERATED_JSON_FILENAMES["test"]
        convert_bioc_to_json([input_xml_path], test_file, include_entities=False)
    else:
        input_file_path = Path(input_file).resolve()  # type: ignore[arg-type]
        if not input_file_path.is_file():
            raise FileNotFoundError(f"Missing input file: {input_file_path}")
        test_file = input_file_path

    model_options = HFModelOptions(
        model_name_or_path=resolve_model_reference(model_path),
        tokenizer_name=str(tokenizer_path) if tokenizer_path is not None else None,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        model_revision=model_revision,
        token=token,
        trust_remote_code=trust_remote_code,
    )

    training_args = _build_prediction_training_args(
        output_dir=output_dir_path,
        per_device_predict_batch_size=per_device_predict_batch_size,
        fp16=fp16,
    )
    configure_logging(training_args)

    raw_datasets = load_raw_datasets({"test": str(test_file)}, cache_dir=model_options.cache_dir)
    inferred_columns = infer_columns(raw_datasets, text_column_name, None)
    columns = DatasetColumns(
        text=inferred_columns.text,
        entities=None,
        document_id=inferred_columns.document_id,
        passage_id=inferred_columns.passage_id,
        passage_offset=inferred_columns.passage_offset,
        record_id=inferred_columns.record_id,
    )

    config = load_model_config(model_options, task_name="ner")
    label_schema = build_label_schema_from_model(config)
    tokenizer = load_fast_tokenizer(model_options, config)
    model = load_token_classification_model(model_options, config)
    effective_max_seq_length = resolve_effective_max_seq_length(tokenizer, config, max_seq_length)
    LOGGER.info("Using max_seq_length=%s", effective_max_seq_length)

    padding: str | bool = "max_length" if pad_to_max_length else False
    with training_args.main_process_first(desc="tokenize prediction dataset"):
        raw_predict_subset, predict_dataset = preprocess_prediction_dataset(
            raw_dataset=raw_datasets["test"],
            columns=columns,
            tokenizer=tokenizer,
            max_seq_length=effective_max_seq_length,
            stride=doc_stride,
            padding=padding,
            num_proc=preprocessing_num_workers,
            overwrite_cache=overwrite_cache,
            limit=max_predict_samples,
        )

    data_collator = TokenClassificationBatchCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )
    trainer = build_trainer(
        model=model,
        training_args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,
        train_dataset=None,
        eval_dataset=None,
    )

    for warmup_index in range(warmup_runs):
        LOGGER.info("Warmup run %d/%d", warmup_index + 1, warmup_runs)
        _ = trainer.predict(predict_dataset, metric_key_prefix=f"warmup_{warmup_index + 1}")
        synchronize_if_cuda()

    synchronize_if_cuda()
    predict_start_time = time.perf_counter()
    predict_output = trainer.predict(predict_dataset, metric_key_prefix="predict")
    synchronize_if_cuda()
    elapsed = time.perf_counter() - predict_start_time

    trainer.save_metrics("predict", dict(predict_output.metrics))

    prediction_logits = predict_output.predictions
    if isinstance(prediction_logits, tuple):
        prediction_logits = prediction_logits[0]

    prediction_entries = reconstruct_prediction_outputs(
        raw_predict_dataset=raw_predict_subset,
        tokenized_predict_dataset=predict_dataset,
        prediction_logits=np.asarray(prediction_logits),
        label_schema=label_schema,
        tokenizer=tokenizer,
        columns=columns,
    )
    save_prediction_outputs(str(output_dir_path), prediction_entries)

    runtime_summary_path = output_dir_path / RUNTIME_SUMMARY_FILENAME
    runtime_summary_path.write_text(
        json.dumps(
            {
                "elapsed_seconds": elapsed,
                "num_passages": len(raw_predict_subset),
                "num_prediction_chunks": len(predict_dataset),
                "model_path": str(model_path),
                "input_xml": str(input_xml) if input_xml is not None else None,
                "input_file": str(input_file) if input_file is not None else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if output_xml is not None:
        output_xml_path = Path(output_xml).resolve()
        output_xml_path.parent.mkdir(parents=True, exist_ok=True)
        write_predictions_to_bioc_xml(input_xml_path, output_xml_path, prediction_entries)  # type: ignore[name-defined]
        LOGGER.info("Wrote BioC XML predictions to %s", output_xml_path)

    return 0


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CellExLink offset-based NER prediction.")
    parser.add_argument("--model-path", type=str, required=True, help="Fine-tuned NER model directory or Hugging Face Hub id.")
    parser.add_argument("--tokenizer-path", type=str, default=None, help="Optional tokenizer path if different from model path.")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-xml", type=Path, help="Input BioC XML file to convert and predict.")
    input_group.add_argument("--input-file", type=Path, help="Input JSON/JSONL/CSV file with raw passage text.")

    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where prediction artifacts are written.")
    parser.add_argument("--output-xml", type=Path, default=None, help="Optional output BioC XML path.")
    parser.add_argument("--text-column-name", default=None, help="Optional raw text column name override.")
    parser.add_argument("--max-seq-length", type=int, default=None, help="Optional tokenizer max sequence length.")
    parser.add_argument("--doc-stride", type=int, default=128, help="Overlap in tokens between overflowing chunks.")
    parser.add_argument("--pad-to-max-length", action="store_true", help="Pad every batch item to max length.")
    parser.add_argument("--max-predict-samples", type=int, default=None, help="Optional cap on prediction examples.")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Number of untimed warmup prediction runs.")
    parser.add_argument("--per-device-predict-batch-size", type=int, default=16, help="Prediction batch size per device.")
    parser.add_argument("--preprocessing-num-workers", type=int, default=None, help="Dataset map workers.")
    parser.add_argument("--overwrite-cache", action="store_true", help="Overwrite cached dataset preprocessing.")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--model-revision", default="main", help="Model revision to use when loading from the Hub.")
    parser.add_argument("--token", default=None, help="Optional Hugging Face auth token.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom code from the model repo.")
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 prediction when supported by hardware.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    return predict_ner(
        model_path=args.model_path,
        input_xml=args.input_xml,
        input_file=args.input_file,
        output_dir=args.output_dir,
        output_xml=args.output_xml,
        tokenizer_path=args.tokenizer_path,
        text_column_name=args.text_column_name,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        pad_to_max_length=args.pad_to_max_length,
        max_predict_samples=args.max_predict_samples,
        warmup_runs=args.warmup_runs,
        per_device_predict_batch_size=args.per_device_predict_batch_size,
        preprocessing_num_workers=args.preprocessing_num_workers,
        overwrite_cache=args.overwrite_cache,
        cache_dir=args.cache_dir,
        model_revision=args.model_revision,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
        fp16=args.fp16,
    )


if __name__ == "__main__":
    raise SystemExit(main())
