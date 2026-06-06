"""
Fine-tuning utilities for the CellExLink normalization encoder.

The final CellExLink paper describes a SapBERT-based linker adapted with
concept-labeled Cell Ontology aliases and CellLink training entities. This file
keeps that training workflow separate from inference code.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from .ontology import load_cell_ontology_terms

DEFAULT_BASE_NEN_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"


@dataclass(slots=True)
class NENTrainingConfig:
    model_name_or_path: str = DEFAULT_BASE_NEN_MODEL
    train_pairs: Optional[str] = None
    ontology_jsonl: Optional[str] = None
    output_dir: str = "models/CellExLink-Sapbert"
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    max_pairs_per_concept: int = 50
    seed: int = 13
    show_progress_bar: bool = True


def read_positive_pairs(path: str | Path) -> list[tuple[str, str]]:
    """
    Read positive NEN training pairs.

    Supported formats:
    - TSV with at least two columns: text_a<TAB>text_b
    - JSONL with keys text_a/text_b, sentence1/sentence2, or left/right
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Training pair file does not exist: {path}")

    pairs: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            if path.suffix == ".jsonl":
                record = json.loads(line)
                left = (
                    record.get("text_a")
                    or record.get("sentence1")
                    or record.get("left")
                    or record.get("name1")
                )
                right = (
                    record.get("text_b")
                    or record.get("sentence2")
                    or record.get("right")
                    or record.get("name2")
                )
                if not left or not right:
                    raise ValueError(f"Missing pair fields on line {line_no} of {path}")
                pairs.append((str(left), str(right)))
            else:
                fields = line.split("\t")
                if len(fields) < 2:
                    raise ValueError(f"Expected at least two TSV columns on line {line_no} of {path}")
                # Skip a common header row.
                if line_no == 1 and fields[0].lower() in {"text_a", "sentence1", "left"}:
                    continue
                pairs.append((fields[0], fields[1]))

    return pairs


def build_positive_pairs_from_ontology(
    ontology_jsonl: str | Path,
    *,
    max_pairs_per_concept: int = 50,
    seed: int = 13,
) -> list[tuple[str, str]]:
    """
    Create positive synonym/name pairs from the Cell Ontology JSONL resource.
    """
    _, concept_metadata = load_cell_ontology_terms(ontology_jsonl)
    rng = random.Random(seed)
    pairs: list[tuple[str, str]] = []

    for metadata in concept_metadata.values():
        names = sorted({metadata.preferred_label, *metadata.names, *metadata.synonyms})
        names = [name for name in names if str(name).strip()]
        if len(names) < 2:
            continue
        concept_pairs = list(itertools.combinations(names, 2))
        if len(concept_pairs) > max_pairs_per_concept:
            concept_pairs = rng.sample(concept_pairs, k=max_pairs_per_concept)
        pairs.extend(concept_pairs)

    rng.shuffle(pairs)
    return pairs


def write_positive_pairs(
    pairs: Iterable[tuple[str, str]],
    output_tsv: str | Path,
) -> Path:
    """Write positive NEN pairs to TSV."""
    output_tsv = Path(output_tsv)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with output_tsv.open("w", encoding="utf-8") as handle:
        handle.write("text_a\ttext_b\n")
        for left, right in pairs:
            handle.write(f"{left}\t{right}\n")
    return output_tsv


def load_training_pairs(config: NENTrainingConfig) -> list[tuple[str, str]]:
    """Load or generate positive NEN training pairs."""
    pairs: list[tuple[str, str]] = []

    if config.train_pairs:
        pairs.extend(read_positive_pairs(config.train_pairs))

    if config.ontology_jsonl:
        pairs.extend(
            build_positive_pairs_from_ontology(
                config.ontology_jsonl,
                max_pairs_per_concept=config.max_pairs_per_concept,
                seed=config.seed,
            )
        )

    if not pairs:
        raise ValueError("Provide --train-pairs and/or --ontology-jsonl.")

    # Deduplicate while preserving order.
    seen: set[tuple[str, str]] = set()
    deduped: list[tuple[str, str]] = []
    for left, right in pairs:
        key = (left.strip(), right.strip())
        rev = (right.strip(), left.strip())
        if not key[0] or not key[1] or key in seen or rev in seen:
            continue
        deduped.append(key)
        seen.add(key)

    return deduped


def train_nen(
    *,
    model_name_or_path: str = DEFAULT_BASE_NEN_MODEL,
    output_dir: str | Path = "models/CellExLink-Sapbert",
    train_pairs: str | Path | None = None,
    ontology_jsonl: str | Path | None = None,
    epochs: int = 1,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    warmup_steps: int = 100,
    max_pairs_per_concept: int = 50,
    seed: int = 13,
    show_progress_bar: bool = True,
) -> Path:
    """
    Fine-tune a sentence-transformers/SapBERT-style encoder for NEN.

    This uses MultipleNegativesRankingLoss, which is a common setup for synonym
    and entity-linking representation learning.
    """
    config = NENTrainingConfig(
        model_name_or_path=str(model_name_or_path),
        train_pairs=str(train_pairs) if train_pairs is not None else None,
        ontology_jsonl=str(ontology_jsonl) if ontology_jsonl is not None else None,
        output_dir=str(output_dir),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        max_pairs_per_concept=max_pairs_per_concept,
        seed=seed,
        show_progress_bar=show_progress_bar,
    )

    pairs = load_training_pairs(config)

    try:
        import torch
        from sentence_transformers import InputExample, SentenceTransformer, losses
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Training requires sentence-transformers and torch. "
            "Install the training extra before running train_nen."
        ) from exc

    random.seed(seed)
    torch.manual_seed(seed)

    model = SentenceTransformer(config.model_name_or_path)
    train_examples = [InputExample(texts=[left, right]) for left, right in pairs]
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=config.batch_size,
    )
    train_loss = losses.MultipleNegativesRankingLoss(model)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=config.epochs,
        warmup_steps=config.warmup_steps,
        optimizer_params={"lr": config.learning_rate},
        output_path=str(output_dir),
        show_progress_bar=config.show_progress_bar,
    )

    metadata = {
        "base_model": config.model_name_or_path,
        "num_training_pairs": len(pairs),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "warmup_steps": config.warmup_steps,
    }
    (output_dir / "cellexlink_nen_training_config.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return output_dir


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the CellExLink normalization encoder.")
    parser.add_argument("--model-name-or-path", default=DEFAULT_BASE_NEN_MODEL)
    parser.add_argument("--train-pairs", default=None, help="TSV/JSONL positive synonym pairs.")
    parser.add_argument("--ontology-jsonl", default=None, help="Cell Ontology JSONL for generated pairs.")
    parser.add_argument("--output-dir", default="models/CellExLink-Sapbert")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-pairs-per-concept", type=int, default=50)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument(
        "--write-generated-pairs",
        default=None,
        help="Optional TSV path for generated/loaded positive pairs without training.",
    )
    return parser.parse_args(argv)


def cli_main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if args.write_generated_pairs:
        config = NENTrainingConfig(
            model_name_or_path=args.model_name_or_path,
            train_pairs=args.train_pairs,
            ontology_jsonl=args.ontology_jsonl,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            max_pairs_per_concept=args.max_pairs_per_concept,
            seed=args.seed,
            show_progress_bar=not args.no_progress_bar,
        )
        pairs = load_training_pairs(config)
        write_positive_pairs(pairs, args.write_generated_pairs)
        print(f"Wrote {len(pairs)} positive pairs to {args.write_generated_pairs}")
        return 0

    output_dir = train_nen(
        model_name_or_path=args.model_name_or_path,
        train_pairs=args.train_pairs,
        ontology_jsonl=args.ontology_jsonl,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_pairs_per_concept=args.max_pairs_per_concept,
        seed=args.seed,
        show_progress_bar=not args.no_progress_bar,
    )
    print(f"Saved CellExLink NEN model to {output_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
