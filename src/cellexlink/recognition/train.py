from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Optional

from transformers import TrainingArguments
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from .bioc import convert_bioc_to_json
from .hf_utils import (
    HFModelOptions,
    load_fast_tokenizer,
    load_model_config,
    load_token_classification_model,
    resolve_effective_max_seq_length,
    resolve_model_reference,
    set_random_seed,
)
from .ner_common import (
    GENERATED_JSON_FILENAMES,
    TokenClassificationBatchCollator,
    build_compute_metrics,
    build_label_schema,
    build_trainer,
    configure_logging,
    infer_columns,
    infer_last_checkpoint,
    load_raw_datasets,
    maybe_log_and_save_metrics,
    preprocess_train_dataset,
)

check_min_version("4.37.0")
require_version("datasets>=1.8.0", "Please install a compatible `datasets` version for CellExLink.")

LOGGER = logging.getLogger(__name__)
PathLike = str | Path


def _prepare_split_file(
    *,
    split_name: str,
    xml_path: Optional[PathLike],
    file_path: Optional[PathLike],
    output_dir: Path,
    overlap_policy: str,
) -> Optional[str]:
    """Return a JSON/JSONL/CSV file for a split, converting BioC XML if needed."""
    if xml_path is not None and file_path is not None:
        raise ValueError(f"Provide only one of `{split_name}_xml` or `{split_name}_file`.")
    if xml_path is None and file_path is None:
        return None

    if xml_path is not None:
        xml_path = Path(xml_path).resolve()
        if not xml_path.is_file():
            raise FileNotFoundError(f"Missing {split_name} XML file: {xml_path}")
        json_path = output_dir / GENERATED_JSON_FILENAMES[split_name]
        convert_bioc_to_json([xml_path], json_path, include_entities=True, overlap_policy=overlap_policy)
        return str(json_path)

    file_path = Path(file_path).resolve()  # type: ignore[arg-type]
    if not file_path.is_file():
        raise FileNotFoundError(f"Missing {split_name} file: {file_path}")
    return str(file_path)


def _build_training_args(
    *,
    output_dir: PathLike,
    do_eval: bool,
    do_predict: bool,
    learning_rate: float,
    num_train_epochs: float,
    weight_decay: float,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    logging_steps: int,
    save_steps: int,
    save_total_limit: Optional[int],
    seed: int,
    fp16: bool,
    overwrite_output_dir: bool,
) -> TrainingArguments:
    """
    Create TrainingArguments with conservative defaults.

    We evaluate manually after training instead of depending on version-specific
    evaluation_strategy/eval_strategy behavior.
    """
    return TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        do_eval=do_eval,
        do_predict=do_predict,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        seed=seed,
        fp16=fp16,
        overwrite_output_dir=overwrite_output_dir,
        remove_unused_columns=False,
        report_to=[],
    )


def train_ner(
    *,
    model_path: PathLike,
    output_dir: PathLike,
    train_xml: Optional[PathLike] = None,
    train_file: Optional[PathLike] = None,
    validation_xml: Optional[PathLike] = None,
    validation_file: Optional[PathLike] = None,
    test_xml: Optional[PathLike] = None,
    test_file: Optional[PathLike] = None,
    config_path: Optional[PathLike] = None,
    tokenizer_path: Optional[PathLike] = None,
    text_column_name: Optional[str] = None,
    entities_column_name: Optional[str] = None,
    max_seq_length: Optional[int] = None,
    doc_stride: int = 128,
    pad_to_max_length: bool = False,
    alignment_mode: str = "expand",
    overlap_policy: str = "last",
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
    max_predict_samples: Optional[int] = None,
    learning_rate: float = 5e-5,
    num_train_epochs: float = 3.0,
    weight_decay: float = 0.0,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 16,
    logging_steps: int = 50,
    save_steps: int = 500,
    save_total_limit: Optional[int] = 2,
    preprocessing_num_workers: Optional[int] = None,
    overwrite_cache: bool = False,
    overwrite_output_dir: bool = False,
    cache_dir: Optional[PathLike] = None,
    model_revision: str = "main",
    token: Optional[str] = None,
    trust_remote_code: bool = False,
    ignore_mismatched_sizes: bool = False,
    fp16: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Fine-tune a token-classification model for CellExLink cell-type recognition.

    Training input can be BioC XML or JSON/JSONL/CSV. JSON/JSONL records should
    contain at least:

        text: str
        entities: list[{"start": int, "end": int, "label": str}]

    Entity offsets are local to the text field.
    """
    if (train_xml is None) == (train_file is None):
        raise ValueError("Provide exactly one of `train_xml` or `train_file`.")
    if doc_stride < 0:
        raise ValueError("`doc_stride` must be >= 0.")
    if alignment_mode not in {"strict", "expand", "skip"}:
        raise ValueError("alignment_mode must be one of: strict, expand, skip.")
    if overlap_policy not in {"last", "error"}:
        raise ValueError("overlap_policy must be one of: last, error.")

    set_random_seed(seed)

    output_dir_path = Path(output_dir).resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)
    converted_dir = output_dir_path / "converted_inputs"
    converted_dir.mkdir(parents=True, exist_ok=True)

    data_files: dict[str, str] = {}
    train_data_file = _prepare_split_file(
        split_name="train",
        xml_path=train_xml,
        file_path=train_file,
        output_dir=converted_dir,
        overlap_policy=overlap_policy,
    )
    assert train_data_file is not None
    data_files["train"] = train_data_file

    validation_data_file = _prepare_split_file(
        split_name="validation",
        xml_path=validation_xml,
        file_path=validation_file,
        output_dir=converted_dir,
        overlap_policy=overlap_policy,
    )
    if validation_data_file is not None:
        data_files["validation"] = validation_data_file

    test_data_file = _prepare_split_file(
        split_name="test",
        xml_path=test_xml,
        file_path=test_file,
        output_dir=converted_dir,
        overlap_policy=overlap_policy,
    )
    if test_data_file is not None:
        data_files["test"] = test_data_file

    model_options = HFModelOptions(
        model_name_or_path=resolve_model_reference(model_path),
        config_name=str(config_path) if config_path is not None else None,
        tokenizer_name=str(tokenizer_path) if tokenizer_path is not None else None,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        model_revision=model_revision,
        token=token,
        trust_remote_code=trust_remote_code,
        ignore_mismatched_sizes=ignore_mismatched_sizes,
    )

    training_args = _build_training_args(
        output_dir=output_dir_path,
        do_eval="validation" in data_files,
        do_predict="test" in data_files,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        seed=seed,
        fp16=fp16,
        overwrite_output_dir=overwrite_output_dir,
    )
    configure_logging(training_args)

    raw_datasets = load_raw_datasets(data_files, cache_dir=model_options.cache_dir)
    columns = infer_columns(raw_datasets, text_column_name, entities_column_name)
    if columns.entities is None:
        raise ValueError(
            "Could not find an entities column. Provide `entities_column_name` or use BioC XML conversion."
        )

    label_schema = build_label_schema(
        raw_datasets,
        entities_column=columns.entities,
        source_splits=["train"],
        overlap_policy=overlap_policy,
    )

    id2label = {index: label for index, label in enumerate(label_schema.label_list)}
    label2id = {label: index for index, label in enumerate(label_schema.label_list)}
    config = load_model_config(
        model_options,
        task_name="ner",
        num_labels=len(label_schema.label_list),
        id2label=id2label,
        label2id=label2id,
    )
    tokenizer = load_fast_tokenizer(model_options, config)
    model = load_token_classification_model(model_options, config)

    effective_max_seq_length = resolve_effective_max_seq_length(tokenizer, config, max_seq_length)
    LOGGER.info("Using max_seq_length=%s", effective_max_seq_length)
    padding: str | bool = "max_length" if pad_to_max_length else False

    with training_args.main_process_first(desc="tokenize training dataset"):
        _, train_dataset = preprocess_train_dataset(
            raw_dataset=raw_datasets["train"],
            columns=columns,
            tokenizer=tokenizer,
            label_schema=label_schema,
            max_seq_length=effective_max_seq_length,
            stride=doc_stride,
            padding=padding,
            alignment_mode=alignment_mode,
            overlap_policy=overlap_policy,
            num_proc=preprocessing_num_workers,
            overwrite_cache=overwrite_cache,
            limit=max_train_samples,
        )

        eval_dataset = None
        if "validation" in raw_datasets:
            _, eval_dataset = preprocess_train_dataset(
                raw_dataset=raw_datasets["validation"],
                columns=columns,
                tokenizer=tokenizer,
                label_schema=label_schema,
                max_seq_length=effective_max_seq_length,
                stride=doc_stride,
                padding=padding,
                alignment_mode=alignment_mode,
                overlap_policy=overlap_policy,
                num_proc=preprocessing_num_workers,
                overwrite_cache=overwrite_cache,
                limit=max_eval_samples,
            )

        predict_dataset = None
        if "test" in raw_datasets:
            _, predict_dataset = preprocess_train_dataset(
                raw_dataset=raw_datasets["test"],
                columns=columns,
                tokenizer=tokenizer,
                label_schema=label_schema,
                max_seq_length=effective_max_seq_length,
                stride=doc_stride,
                padding=padding,
                alignment_mode=alignment_mode,
                overlap_policy=overlap_policy,
                num_proc=preprocessing_num_workers,
                overwrite_cache=overwrite_cache,
                limit=max_predict_samples,
            )

    data_collator = TokenClassificationBatchCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )
    compute_metrics = build_compute_metrics(label_schema)
    trainer = build_trainer(
        model=model,
        training_args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    last_checkpoint = infer_last_checkpoint(training_args)
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(str(output_dir_path))
    tokenizer.save_pretrained(str(output_dir_path))

    train_metrics = dict(train_result.metrics)
    train_metrics["train_samples"] = len(raw_datasets["train"])
    maybe_log_and_save_metrics(trainer, "train", train_metrics)
    trainer.save_state()

    label_schema_path = output_dir_path / "label_schema.json"
    label_schema_path.write_text(
        json.dumps(
            {
                "label_list": label_schema.label_list,
                "label_to_id": label_schema.label_to_id,
                "id_to_label": {str(k): v for k, v in label_schema.id_to_label.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    results: dict[str, Any] = {
        "output_dir": str(output_dir_path),
        "label_schema": str(label_schema_path),
        "train_metrics": train_metrics,
    }

    if eval_dataset is not None:
        eval_metrics = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix="eval")
        eval_metrics["eval_samples"] = len(raw_datasets["validation"])
        maybe_log_and_save_metrics(trainer, "eval", eval_metrics)
        results["eval_metrics"] = dict(eval_metrics)

    if predict_dataset is not None:
        predict_output = trainer.predict(predict_dataset, metric_key_prefix="test")
        test_metrics = dict(predict_output.metrics)
        test_metrics["test_samples"] = len(raw_datasets["test"])
        maybe_log_and_save_metrics(trainer, "test", test_metrics)
        results["test_metrics"] = test_metrics

    summary_path = output_dir_path / "train_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    results["summary"] = str(summary_path)
    return results


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune CellExLink's cell-type NER model.")
    parser.add_argument("--model-path", required=True, help="Base model or checkpoint path/Hub id, e.g. Bioformer or PubMedBERT.")
    parser.add_argument("--output-dir", required=True, help="Directory where the fine-tuned model will be saved.")

    train_group = parser.add_mutually_exclusive_group(required=True)
    train_group.add_argument("--train-xml", default=None, help="Training BioC XML file or directory.")
    train_group.add_argument("--train-file", default=None, help="Training JSON/JSONL/CSV file.")

    validation_group = parser.add_mutually_exclusive_group(required=False)
    validation_group.add_argument("--validation-xml", default=None, help="Validation BioC XML file or directory.")
    validation_group.add_argument("--validation-file", default=None, help="Validation JSON/JSONL/CSV file.")

    test_group = parser.add_mutually_exclusive_group(required=False)
    test_group.add_argument("--test-xml", default=None, help="Test BioC XML file or directory.")
    test_group.add_argument("--test-file", default=None, help="Test JSON/JSONL/CSV file.")

    parser.add_argument("--config-path", default=None, help="Optional config path if different from model path.")
    parser.add_argument("--tokenizer-path", default=None, help="Optional tokenizer path if different from model path.")
    parser.add_argument("--text-column-name", default=None, help="Text column name override.")
    parser.add_argument("--entities-column-name", default=None, help="Entities column name override.")
    parser.add_argument("--max-seq-length", type=int, default=None, help="Optional tokenizer max sequence length.")
    parser.add_argument("--doc-stride", type=int, default=128, help="Overlap in tokens between overflow chunks.")
    parser.add_argument("--pad-to-max-length", action="store_true", help="Pad all examples to max_seq_length.")
    parser.add_argument("--alignment-mode", choices=["strict", "expand", "skip"], default="expand")
    parser.add_argument("--overlap-policy", choices=["last", "error"], default="last")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--max-predict-samples", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=16)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--preprocessing-num-workers", type=int, default=None)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--token", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--ignore-mismatched-sizes", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    results = train_ner(
        model_path=args.model_path,
        output_dir=args.output_dir,
        train_xml=args.train_xml,
        train_file=args.train_file,
        validation_xml=args.validation_xml,
        validation_file=args.validation_file,
        test_xml=args.test_xml,
        test_file=args.test_file,
        config_path=args.config_path,
        tokenizer_path=args.tokenizer_path,
        text_column_name=args.text_column_name,
        entities_column_name=args.entities_column_name,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        pad_to_max_length=args.pad_to_max_length,
        alignment_mode=args.alignment_mode,
        overlap_policy=args.overlap_policy,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_predict_samples=args.max_predict_samples,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        preprocessing_num_workers=args.preprocessing_num_workers,
        overwrite_cache=args.overwrite_cache,
        overwrite_output_dir=args.overwrite_output_dir,
        cache_dir=args.cache_dir,
        model_revision=args.model_revision,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        fp16=args.fp16,
        seed=args.seed,
    )
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
