from __future__ import annotations

import inspect
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence

import datasets
import numpy as np
import transformers
from datasets import Dataset, DatasetDict, load_dataset
from transformers import DataCollatorForTokenClassification, PreTrainedTokenizerFast, Trainer, TrainingArguments

LOGGER = logging.getLogger(__name__)

IGNORE_INDEX = -100
INTERNAL_EXAMPLE_INDEX_COLUMN = "__example_index__"
GENERATED_JSON_FILENAMES = {
    "train": "train.hf.jsonl",
    "validation": "validation.hf.jsonl",
    "test": "test.hf.jsonl",
}


@dataclass(slots=True)
class DatasetColumns:
    """Column names used by the offset-based NER pipeline."""

    text: str
    entities: Optional[str]
    document_id: Optional[str]
    passage_id: Optional[str]
    passage_offset: Optional[str]
    record_id: Optional[str]


@dataclass(slots=True)
class LabelSchema:
    """BIO label mappings for token classification."""

    label_list: list[str]
    label_to_id: dict[str, int]
    id_to_label: dict[int, str]

    def encode(self, label: str) -> int:
        try:
            return self.label_to_id[str(label)]
        except KeyError as exc:
            raise KeyError(f"Unknown label {label!r}. Known labels: {self.label_list}") from exc


@dataclass(frozen=True, slots=True)
class Entity:
    """A local-offset entity span used for BIO label generation."""

    start: int
    end: int
    label: str

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"Invalid entity span: {self.start}..{self.end}")


class TokenClassificationBatchCollator(DataCollatorForTokenClassification):
    """Drop metadata fields before padding/model dispatch."""

    ignored_feature_keys = {
        "sample_index",
        "offset_mapping",
        "special_tokens_mask",
        "text",
        "document_id",
        "passage_id",
        "passage_offset",
        "id",
    }

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        sanitized = [
            {key: value for key, value in feature.items() if key not in self.ignored_feature_keys}
            for feature in features
        ]
        return super().__call__(sanitized)


def configure_logging(training_args: TrainingArguments) -> None:
    """Configure logging consistently with Hugging Face Trainer."""
    import sys

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    LOGGER.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    LOGGER.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, fp16: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        training_args.parallel_mode.value == "distributed",
        training_args.fp16,
    )
    LOGGER.info("Training arguments: %s", training_args)


def load_raw_datasets(data_files: dict[str, str], cache_dir: Optional[str]) -> DatasetDict:
    """Load JSONL/JSON/CSV data files into a DatasetDict."""
    if not data_files:
        raise ValueError("No dataset files were provided.")

    first_path = next(iter(data_files.values()))
    extension = first_path.rsplit(".", 1)[-1].lower()
    if extension == "jsonl":
        extension = "json"
    if extension not in {"json", "csv"}:
        raise ValueError(f"Unsupported dataset extension: {extension}. Use JSON, JSONL, or CSV.")

    return load_dataset(extension, data_files=data_files, cache_dir=cache_dir)


def choose_reference_split(raw_datasets: DatasetDict) -> str:
    for split_name in ("train", "validation", "test"):
        if split_name in raw_datasets:
            return split_name
    raise ValueError("No dataset splits were loaded.")


def infer_columns(
    raw_datasets: DatasetDict,
    text_column_name: Optional[str],
    entities_column_name: Optional[str],
) -> DatasetColumns:
    """Infer text/entity/metadata columns from the first available split."""
    reference_split = choose_reference_split(raw_datasets)
    column_names = raw_datasets[reference_split].column_names

    if text_column_name is not None:
        text_column = text_column_name
    elif "text" in column_names:
        text_column = "text"
    else:
        text_column = column_names[0]

    if entities_column_name is not None:
        entities_column = entities_column_name
    elif "entities" in column_names:
        entities_column = "entities"
    else:
        entities_column = None

    return DatasetColumns(
        text=text_column,
        entities=entities_column,
        document_id="document_id" if "document_id" in column_names else None,
        passage_id="passage_id" if "passage_id" in column_names else None,
        passage_offset="passage_offset" if "passage_offset" in column_names else None,
        record_id="id" if "id" in column_names else None,
    )


def entities_overlap(left: Entity, right: Entity) -> bool:
    return max(left.start, right.start) < min(left.end, right.end)


def validate_no_overlapping_entities(entities: Sequence[Entity]) -> None:
    for index, left in enumerate(entities):
        for right in entities[index + 1 :]:
            if entities_overlap(left, right):
                raise ValueError(
                    "Overlapping entities are not representable in flat BIO tagging when overlap_policy='error': "
                    f"{left.label!r} at {left.start}-{left.end} overlaps "
                    f"{right.label!r} at {right.start}-{right.end}."
                )


def flatten_entities_for_flat_ner(
    entities: Sequence[Entity],
    *,
    overlap_policy: str = "last",
) -> list[Entity]:
    """Validate entity overlaps for a flat BIO tagging setup."""
    if overlap_policy not in {"last", "error"}:
        raise ValueError("overlap_policy must be one of: last, error.")
    entities_list = list(entities)
    if overlap_policy == "error":
        validate_no_overlapping_entities(entities_list)
    return entities_list


def canonicalize_entities(raw_value: Any, *, overlap_policy: str = "last") -> list[Entity]:
    """Convert a dataset `entities` value into Entity objects."""
    if raw_value is None:
        return []

    value = raw_value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        value = json.loads(stripped)

    if not isinstance(value, list):
        raise ValueError(f"Expected `entities` to be a list. Got: {type(value).__name__}")

    entities: list[Entity] = []
    for item in value:
        if isinstance(item, Entity):
            entities.append(item)
            continue
        if not isinstance(item, dict):
            raise ValueError(f"Expected entity record to be a dict. Got: {item!r}")
        if "start" not in item or "end" not in item or "label" not in item:
            raise ValueError(f"Entity record is missing required keys: {item!r}")
        entities.append(Entity(start=int(item["start"]), end=int(item["end"]), label=str(item["label"])))

    return flatten_entities_for_flat_ner(entities, overlap_policy=overlap_policy)


def label_sort_key(label: str) -> tuple[int, str]:
    if label == "O":
        return (0, label)
    if label.startswith("B-"):
        return (1, label[2:] + "\t0")
    if label.startswith("I-"):
        return (1, label[2:] + "\t1")
    return (2, label)


def build_label_schema(
    raw_datasets: DatasetDict,
    entities_column: str,
    source_splits: Sequence[str],
    *,
    overlap_policy: str = "last",
) -> LabelSchema:
    """Build BIO label mappings from training entities."""
    entity_types: set[str] = set()
    for split_name in source_splits:
        if split_name not in raw_datasets:
            continue
        split_dataset = raw_datasets[split_name]
        if entities_column not in split_dataset.column_names:
            continue
        for raw_entities in split_dataset[entities_column]:
            for entity in canonicalize_entities(raw_entities, overlap_policy=overlap_policy):
                entity_types.add(entity.label)

    if not entity_types:
        raise ValueError(f"Could not infer labels from entity column `{entities_column}`.")

    label_list = ["O"]
    for entity_type in sorted(entity_types):
        label_list.append(f"B-{entity_type}")
        label_list.append(f"I-{entity_type}")
    label_list = sorted(label_list, key=label_sort_key)

    label_to_id = {label: index for index, label in enumerate(label_list)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    return LabelSchema(label_list=label_list, label_to_id=label_to_id, id_to_label=id_to_label)


def canonicalize_id2label_mapping(id2label: dict[Any, str]) -> dict[int, str]:
    return {int(index): str(label) for index, label in id2label.items()}


def build_label_schema_from_model(config: transformers.PretrainedConfig) -> LabelSchema:
    """Read BIO label mappings from a saved model config."""
    if not getattr(config, "id2label", None):
        raise ValueError("Prediction requires label mappings in the saved model config.")

    canonical = canonicalize_id2label_mapping(config.id2label)
    label_list = [canonical[index] for index in range(len(canonical))]
    label_to_id = {label: index for index, label in enumerate(label_list)}
    return LabelSchema(
        label_list=label_list,
        label_to_id=label_to_id,
        id_to_label={index: label for label, index in label_to_id.items()},
    )


def maybe_align_label_schema_to_model(label_schema: LabelSchema, config: transformers.PretrainedConfig) -> LabelSchema:
    """Use the existing model label order when compatible."""
    existing_label2id = {str(label): int(index) for label, index in getattr(config, "label2id", {}).items()}
    default_label2id = transformers.PretrainedConfig(num_labels=len(label_schema.label_list)).label2id

    if not existing_label2id or existing_label2id == default_label2id:
        return label_schema

    if sorted(existing_label2id.keys()) != sorted(label_schema.label_list):
        LOGGER.warning(
            "Model label names do not match training label names. Continuing with training label order. "
            "Model labels: %s; training labels: %s",
            sorted(existing_label2id.keys()),
            sorted(label_schema.label_list),
        )
        return label_schema

    canonical = canonicalize_id2label_mapping(getattr(config, "id2label", {}))
    aligned_label_list = [canonical[index] for index in range(len(canonical))]
    LOGGER.info("Using label order from loaded model config: %s", aligned_label_list)
    return LabelSchema(
        label_list=aligned_label_list,
        label_to_id={label: index for index, label in enumerate(aligned_label_list)},
        id_to_label={index: label for index, label in enumerate(aligned_label_list)},
    )


def select_subset(dataset: Dataset, limit: Optional[int]) -> Dataset:
    if limit is None:
        return dataset
    return dataset.select(range(min(len(dataset), limit)))


def attach_example_indices(dataset: Dataset) -> Dataset:
    if INTERNAL_EXAMPLE_INDEX_COLUMN in dataset.column_names:
        return dataset
    return dataset.add_column(INTERNAL_EXAMPLE_INDEX_COLUMN, list(range(len(dataset))))


def full_tokenize_texts(texts: Sequence[str], tokenizer: PreTrainedTokenizerFast) -> list[list[tuple[int, int]]]:
    encodings = tokenizer(
        list(texts),
        add_special_tokens=False,
        return_offsets_mapping=True,
        padding=False,
        truncation=False,
    )
    return [[tuple(map(int, offset)) for offset in sample_offsets] for sample_offsets in encodings["offset_mapping"]]


def build_full_token_label_map(
    offsets: Sequence[tuple[int, int]],
    entities: Sequence[Entity],
    label_schema: LabelSchema,
    alignment_mode: str,
    *,
    overlap_policy: str = "last",
) -> dict[tuple[int, int], int]:
    """Assign BIO labels to full-document tokenizer offsets."""
    if alignment_mode not in {"strict", "expand", "skip"}:
        raise ValueError(f"Unsupported alignment_mode={alignment_mode!r}. Use one of: strict, expand, skip.")
    if overlap_policy not in {"last", "error"}:
        raise ValueError("overlap_policy must be one of: last, error.")

    owner_by_offset: dict[tuple[int, int], str] = {}
    owner_label: dict[str, str] = {}
    normalized_entities = flatten_entities_for_flat_ner(entities, overlap_policy=overlap_policy)

    for entity_index, entity in enumerate(normalized_entities):
        covered_offsets: list[tuple[int, int]] = []
        misaligned = False

        for token_start, token_end in offsets:
            if token_end <= token_start:
                continue
            overlap = min(token_end, entity.end) - max(token_start, entity.start)
            if overlap <= 0:
                continue
            fully_inside = token_start >= entity.start and token_end <= entity.end

            if alignment_mode == "expand":
                covered_offsets.append((token_start, token_end))
            elif fully_inside:
                covered_offsets.append((token_start, token_end))
            else:
                misaligned = True

        if alignment_mode == "strict" and misaligned:
            raise ValueError(f"Entity {entity.label!r} at {entity.start}-{entity.end} does not align to tokenizer offsets.")
        if alignment_mode == "skip" and misaligned:
            continue
        if not covered_offsets:
            if alignment_mode == "strict":
                raise ValueError(f"Entity {entity.label!r} at {entity.start}-{entity.end} does not align to any tokenizer offsets.")
            continue

        owner_id = f"{entity_index}:{entity.label}:{entity.start}:{entity.end}"
        owner_label[owner_id] = entity.label

        if overlap_policy == "error":
            conflicting = [offset for offset in covered_offsets if offset in owner_by_offset and owner_by_offset[offset] != owner_id]
            if conflicting:
                first = conflicting[0]
                raise ValueError(
                    "Overlapping entities are not representable in flat BIO tagging when overlap_policy='error': "
                    f"token span {first} receives multiple entities."
                )

        for offset in covered_offsets:
            owner_by_offset[offset] = owner_id

    label_map: dict[tuple[int, int], int] = {}
    previous_owner_id: Optional[str] = None
    for token_start, token_end in offsets:
        if token_end <= token_start:
            previous_owner_id = None
            continue
        offset = (token_start, token_end)
        owner_id = owner_by_offset.get(offset)
        if owner_id is None:
            previous_owner_id = None
            continue
        prefix = "I-" if owner_id == previous_owner_id else "B-"
        label_map[offset] = label_schema.encode(prefix + owner_label[owner_id])
        previous_owner_id = owner_id

    return label_map


def tokenize_labeled_examples(
    examples: dict[str, list[Any]],
    *,
    tokenizer: PreTrainedTokenizerFast,
    text_column: str,
    entities_column: str,
    label_schema: LabelSchema,
    max_seq_length: int,
    stride: int,
    padding: str | bool,
    alignment_mode: str,
    overlap_policy: str,
) -> dict[str, list[Any]]:
    """Tokenize labeled examples and create BIO labels for every overflow chunk."""
    texts = [str(text) for text in examples[text_column]]
    full_offsets_batch = full_tokenize_texts(texts, tokenizer=tokenizer)

    full_label_maps: list[dict[tuple[int, int], int]] = []
    for offsets, raw_entities in zip(full_offsets_batch, examples[entities_column]):
        entities = canonicalize_entities(raw_entities, overlap_policy=overlap_policy)
        full_label_maps.append(
            build_full_token_label_map(
                offsets=offsets,
                entities=entities,
                label_schema=label_schema,
                alignment_mode=alignment_mode,
                overlap_policy=overlap_policy,
            )
        )

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_seq_length,
        stride=stride,
        padding=padding,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )
    overflow_to_sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    special_tokens_masks = tokenized.pop("special_tokens_mask")
    offset_mappings = tokenized.pop("offset_mapping")

    labels: list[list[int]] = []
    for encoded_index, source_index in enumerate(overflow_to_sample_mapping):
        chunk_labels: list[int] = []
        full_label_map = full_label_maps[int(source_index)]
        for offset, is_special in zip(offset_mappings[encoded_index], special_tokens_masks[encoded_index]):
            token_start, token_end = int(offset[0]), int(offset[1])
            if is_special or token_end <= token_start:
                chunk_labels.append(IGNORE_INDEX)
            else:
                chunk_labels.append(full_label_map.get((token_start, token_end), label_schema.encode("O")))
        labels.append(chunk_labels)

    tokenized["labels"] = labels
    return tokenized


def tokenize_prediction_examples(
    examples: dict[str, list[Any]],
    *,
    tokenizer: PreTrainedTokenizerFast,
    text_column: str,
    example_index_column: str,
    max_seq_length: int,
    stride: int,
    padding: str | bool,
) -> dict[str, list[Any]]:
    """Tokenize unlabeled examples while preserving source example indices."""
    texts = [str(text) for text in examples[text_column]]
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_seq_length,
        stride=stride,
        padding=padding,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )
    overflow_to_sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    tokenized["sample_index"] = [int(examples[example_index_column][index]) for index in overflow_to_sample_mapping]
    return tokenized


def preprocess_train_dataset(
    raw_dataset: Dataset,
    *,
    columns: DatasetColumns,
    tokenizer: PreTrainedTokenizerFast,
    label_schema: LabelSchema,
    max_seq_length: int,
    stride: int,
    padding: str | bool,
    alignment_mode: str,
    overlap_policy: str,
    num_proc: Optional[int],
    overwrite_cache: bool,
    limit: Optional[int],
) -> tuple[Dataset, Dataset]:
    if columns.entities is None:
        raise ValueError("Training requires an entities column.")

    raw_subset = attach_example_indices(select_subset(raw_dataset, limit))
    tokenized = raw_subset.map(
        lambda batch: tokenize_labeled_examples(
            batch,
            tokenizer=tokenizer,
            text_column=columns.text,
            entities_column=columns.entities,
            label_schema=label_schema,
            max_seq_length=max_seq_length,
            stride=stride,
            padding=padding,
            alignment_mode=alignment_mode,
            overlap_policy=overlap_policy,
        ),
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=not overwrite_cache,
        remove_columns=raw_subset.column_names,
        desc="Tokenizing labeled dataset with raw-text offsets",
    )
    return raw_subset, tokenized


def preprocess_prediction_dataset(
    raw_dataset: Dataset,
    *,
    columns: DatasetColumns,
    tokenizer: PreTrainedTokenizerFast,
    max_seq_length: int,
    stride: int,
    padding: str | bool,
    num_proc: Optional[int],
    overwrite_cache: bool,
    limit: Optional[int],
) -> tuple[Dataset, Dataset]:
    raw_subset = attach_example_indices(select_subset(raw_dataset, limit))
    tokenized = raw_subset.map(
        lambda batch: tokenize_prediction_examples(
            batch,
            tokenizer=tokenizer,
            text_column=columns.text,
            example_index_column=INTERNAL_EXAMPLE_INDEX_COLUMN,
            max_seq_length=max_seq_length,
            stride=stride,
            padding=padding,
        ),
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=not overwrite_cache,
        remove_columns=raw_subset.column_names,
        desc="Tokenizing prediction dataset with raw-text offsets",
    )
    return raw_subset, tokenized


def build_trainer(
    *,
    model: transformers.PreTrainedModel,
    training_args: TrainingArguments,
    tokenizer: PreTrainedTokenizerFast,
    data_collator: DataCollatorForTokenClassification,
    compute_metrics: Optional[Callable[[Any], dict[str, float]]],
    train_dataset: Optional[Dataset],
    eval_dataset: Optional[Dataset],
) -> Trainer:
    """Create a Trainer while supporting old and new Transformers constructor names."""
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    return Trainer(**trainer_kwargs)


def normalize_strategy_name(value: Any) -> str:
    if value is None:
        return "no"
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def has_active_eval_strategy(training_args: TrainingArguments) -> bool:
    strategy = getattr(training_args, "eval_strategy", None)
    if strategy is None:
        strategy = getattr(training_args, "evaluation_strategy", None)
    return normalize_strategy_name(strategy) != "no"


def normalize_training_args_for_train_only(training_args: TrainingArguments, *, has_validation_data: bool) -> None:
    if training_args.load_best_model_at_end and not has_validation_data:
        raise ValueError("`load_best_model_at_end=True` requires validation data.")
    if has_active_eval_strategy(training_args) and not has_validation_data:
        raise ValueError("An evaluation strategy was requested, but no validation dataset was provided.")


def infer_last_checkpoint(training_args: TrainingArguments) -> Optional[str]:
    """Return the latest checkpoint if output_dir already contains one."""
    from transformers.trainer_utils import get_last_checkpoint

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and os.listdir(training_args.output_dir):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite-output-dir to train from scratch."
            )
    return last_checkpoint


def maybe_log_and_save_metrics(trainer: Trainer, split_name: str, metrics: dict[str, Any]) -> None:
    trainer.log_metrics(split_name, metrics)
    trainer.save_metrics(split_name, metrics)


def build_compute_metrics(label_schema: LabelSchema) -> Callable[[Any], dict[str, float]]:
    """Build seqeval metrics when available, otherwise return token accuracy."""

    def compute_metrics(eval_prediction: Any) -> dict[str, float]:
        predictions = eval_prediction.predictions
        labels = eval_prediction.label_ids
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        pred_ids = np.argmax(predictions, axis=2)

        true_predictions: list[list[str]] = []
        true_labels: list[list[str]] = []
        token_correct = 0
        token_total = 0

        for prediction_row, label_row in zip(pred_ids, labels):
            pred_seq: list[str] = []
            label_seq: list[str] = []
            for prediction_id, label_id in zip(prediction_row, label_row):
                if int(label_id) == IGNORE_INDEX:
                    continue
                pred_label = label_schema.id_to_label[int(prediction_id)]
                true_label = label_schema.id_to_label[int(label_id)]
                pred_seq.append(pred_label)
                label_seq.append(true_label)
                token_total += 1
                if pred_label == true_label:
                    token_correct += 1
            true_predictions.append(pred_seq)
            true_labels.append(label_seq)

        metrics: dict[str, float] = {
            "token_accuracy": float(token_correct / token_total) if token_total else 0.0,
        }

        try:
            from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score

            metrics.update(
                {
                    "precision": float(precision_score(true_labels, true_predictions)),
                    "recall": float(recall_score(true_labels, true_predictions)),
                    "f1": float(f1_score(true_labels, true_predictions)),
                    "seqeval_accuracy": float(accuracy_score(true_labels, true_predictions)),
                }
            )
        except Exception:
            LOGGER.debug("seqeval is not installed; reporting token_accuracy only.")

        return metrics

    return compute_metrics


def reconstruct_entities_from_offsets(
    *,
    text: str,
    full_offsets: Sequence[tuple[int, int]],
    predicted_label_ids_by_offset: dict[tuple[int, int], int],
    label_schema: LabelSchema,
    passage_offset: int,
) -> list[dict[str, Any]]:
    """Convert BIO token predictions into entity spans."""
    entities: list[dict[str, Any]] = []
    active_label: Optional[str] = None
    active_start: Optional[int] = None
    active_end: Optional[int] = None

    def close_active() -> None:
        nonlocal active_label, active_start, active_end
        if active_label is None or active_start is None or active_end is None:
            return
        entities.append(
            {
                "label": active_label,
                "start_local": int(active_start),
                "end_local": int(active_end),
                "start": passage_offset + int(active_start),
                "end": passage_offset + int(active_end),
                "text": text[int(active_start) : int(active_end)],
            }
        )
        active_label = None
        active_start = None
        active_end = None

    for token_start, token_end in full_offsets:
        if token_end <= token_start:
            continue

        label_id = predicted_label_ids_by_offset.get((token_start, token_end), label_schema.encode("O"))
        tag = label_schema.id_to_label[int(label_id)]

        if tag == "O":
            close_active()
            continue

        if "-" in tag:
            prefix, entity_label = tag.split("-", 1)
        else:
            prefix, entity_label = "B", tag

        starts_new = prefix == "B" or active_label is None or active_label != entity_label
        if starts_new:
            close_active()
            active_label = entity_label
            active_start = token_start
            active_end = token_end
        else:
            active_end = token_end

    close_active()
    return entities


def reconstruct_prediction_outputs(
    *,
    raw_predict_dataset: Dataset,
    tokenized_predict_dataset: Dataset,
    prediction_logits: Sequence[Sequence[Sequence[float]]],
    label_schema: LabelSchema,
    tokenizer: PreTrainedTokenizerFast,
    columns: DatasetColumns,
) -> list[dict[str, Any]]:
    """Aggregate overflow-window logits and reconstruct full-passage entities."""
    aggregated_logits: dict[int, dict[tuple[int, int], list[float]]] = {}
    aggregated_counts: dict[int, dict[tuple[int, int], int]] = {}

    for chunk_index, chunk_logits in enumerate(prediction_logits):
        sample_index = int(tokenized_predict_dataset[chunk_index]["sample_index"])
        offset_mapping = tokenized_predict_dataset[chunk_index]["offset_mapping"]
        special_tokens_mask = tokenized_predict_dataset[chunk_index]["special_tokens_mask"]

        sample_logits = aggregated_logits.setdefault(sample_index, {})
        sample_counts = aggregated_counts.setdefault(sample_index, {})

        for token_logits, offset, is_special in zip(chunk_logits, offset_mapping, special_tokens_mask):
            token_start, token_end = int(offset[0]), int(offset[1])
            if is_special or token_end <= token_start:
                continue

            key = (token_start, token_end)
            if key not in sample_logits:
                sample_logits[key] = [float(value) for value in token_logits]
                sample_counts[key] = 1
            else:
                sample_logits[key] = [old + float(new) for old, new in zip(sample_logits[key], token_logits)]
                sample_counts[key] += 1

    outputs: list[dict[str, Any]] = []
    for raw_index, example in enumerate(raw_predict_dataset):
        text = str(example[columns.text])
        full_offsets = full_tokenize_texts([text], tokenizer=tokenizer)[0]
        sample_logits = aggregated_logits.get(raw_index, {})
        sample_counts = aggregated_counts.get(raw_index, {})

        predicted_label_ids_by_offset: dict[tuple[int, int], int] = {}
        for offset in full_offsets:
            if offset not in sample_logits:
                continue
            count = max(sample_counts.get(offset, 1), 1)
            averaged = [value / count for value in sample_logits[offset]]
            best_label = max(range(len(averaged)), key=lambda index: averaged[index])
            predicted_label_ids_by_offset[offset] = int(best_label)

        passage_offset = int(example[columns.passage_offset]) if columns.passage_offset and columns.passage_offset in example else 0
        predicted_entities = reconstruct_entities_from_offsets(
            text=text,
            full_offsets=full_offsets,
            predicted_label_ids_by_offset=predicted_label_ids_by_offset,
            label_schema=label_schema,
            passage_offset=passage_offset,
        )

        entry: dict[str, Any] = {
            "id": example[columns.record_id] if columns.record_id and columns.record_id in example else raw_index,
            "text": text,
            "passage_offset": passage_offset,
            "predicted_entities": predicted_entities,
        }
        if columns.document_id and columns.document_id in example:
            entry["document_id"] = example[columns.document_id]
        if columns.passage_id and columns.passage_id in example:
            entry["passage_id"] = example[columns.passage_id]
        outputs.append(entry)

    return outputs


def save_prediction_outputs(output_dir: str, prediction_entries: list[dict[str, Any]]) -> None:
    """Write predictions.json and predictions.jsonl."""
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "predictions.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(prediction_entries, handle, ensure_ascii=False, indent=2)

    jsonl_path = os.path.join(output_dir, "predictions.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for entry in prediction_entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
