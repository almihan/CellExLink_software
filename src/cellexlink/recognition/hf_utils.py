from __future__ import annotations

import logging
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer, PreTrainedTokenizerFast

LOGGER = logging.getLogger(__name__)
PathLike = str | Path

REASONABLE_MAX_LENGTH_FALLBACK = 512
TOKENIZER_TYPES_REQUIRING_PREFIX_SPACE = {"bloom", "deberta", "gpt2", "roberta"}


@dataclass(slots=True)
class HFModelOptions:
    """Configuration for loading Hugging Face token-classification models."""

    model_name_or_path: str
    config_name: Optional[str] = None
    tokenizer_name: Optional[str] = None
    cache_dir: Optional[str] = None
    model_revision: str = "main"
    token: Optional[str] = None
    use_auth_token: Optional[bool] = None
    trust_remote_code: bool = False
    ignore_mismatched_sizes: bool = False


def resolve_model_reference(model_reference: PathLike) -> str:
    """Return a local absolute path when the reference exists, otherwise keep the Hub id."""
    model_reference_str = str(model_reference)
    candidate = Path(model_reference_str)
    if candidate.exists():
        return str(candidate.resolve())
    return model_reference_str


def normalize_auth_arguments(options: HFModelOptions) -> None:
    """Handle the old `use_auth_token` argument while preferring `token`."""
    if options.use_auth_token is not None:
        warnings.warn("`use_auth_token` is deprecated. Use `token` instead.", FutureWarning, stacklevel=2)
        if options.token is not None:
            raise ValueError("`token` and `use_auth_token` were both specified. Please set only `token`.")
        # Transformers accepts bool here for backward compatibility.
        options.token = options.use_auth_token  # type: ignore[assignment]


def redact_command_for_logging(parts: list[str]) -> str:
    """Return a command string with token values redacted."""
    redacted: list[str] = []
    i = 0
    while i < len(parts):
        part = parts[i]
        redacted.append(part)
        if part in {"--token", "--hf-token"} and i + 1 < len(parts):
            redacted.append("***REDACTED***")
            i += 2
            continue
        i += 1
    return " ".join(redacted)


def set_random_seed(seed: Optional[int]) -> None:
    """Set Python, NumPy and Torch seeds when a seed is provided."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - torch may not be available in light tests
        LOGGER.debug("Torch is not available; only Python/NumPy seeds were set.")


def get_torch_device() -> str:
    """Return the preferred torch device name."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # pragma: no cover
        pass
    return "cpu"


def synchronize_if_cuda() -> None:
    """Synchronize CUDA timing when CUDA is available."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:  # pragma: no cover
        return


def configure_transformers_cache(cache_dir: Optional[PathLike]) -> None:
    """Optionally set Hugging Face cache environment variables."""
    if cache_dir is None:
        return
    cache_path = Path(cache_dir).expanduser().resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_path))


def load_model_config(
    options: HFModelOptions,
    *,
    task_name: str = "ner",
    num_labels: Optional[int] = None,
    id2label: Optional[dict[int, str]] = None,
    label2id: Optional[dict[str, int]] = None,
):
    """Load an AutoConfig for token classification."""
    normalize_auth_arguments(options)
    config_kwargs: dict[str, Any] = {
        "finetuning_task": task_name,
        "cache_dir": options.cache_dir,
        "revision": options.model_revision,
        "token": options.token,
        "trust_remote_code": options.trust_remote_code,
    }
    if num_labels is not None:
        config_kwargs["num_labels"] = num_labels
    if id2label is not None:
        config_kwargs["id2label"] = id2label
    if label2id is not None:
        config_kwargs["label2id"] = label2id

    return AutoConfig.from_pretrained(
        options.config_name or options.model_name_or_path,
        **config_kwargs,
    )


def load_fast_tokenizer(options: HFModelOptions, config) -> PreTrainedTokenizerFast:
    """Load a fast tokenizer and fail early if only a slow tokenizer is available."""
    tokenizer_name_or_path = options.tokenizer_name or options.model_name_or_path
    tokenizer_kwargs: dict[str, Any] = {
        "cache_dir": options.cache_dir,
        "use_fast": True,
        "revision": options.model_revision,
        "token": options.token,
        "trust_remote_code": options.trust_remote_code,
    }
    if getattr(config, "model_type", None) in TOKENIZER_TYPES_REQUIRING_PREFIX_SPACE:
        tokenizer_kwargs["add_prefix_space"] = True

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError("CellExLink recognition requires a fast tokenizer for offset-based span alignment.")
    return tokenizer


def load_token_classification_model(options: HFModelOptions, config):
    """Load an AutoModelForTokenClassification."""
    return AutoModelForTokenClassification.from_pretrained(
        options.model_name_or_path,
        from_tf=str(options.model_name_or_path).endswith(".ckpt"),
        config=config,
        cache_dir=options.cache_dir,
        revision=options.model_revision,
        token=options.token,
        trust_remote_code=options.trust_remote_code,
        ignore_mismatched_sizes=options.ignore_mismatched_sizes,
    )


def resolve_effective_max_seq_length(
    tokenizer: PreTrainedTokenizerFast,
    config,
    requested_max_seq_length: Optional[int],
) -> int:
    """Choose a safe max sequence length for tokenizer overflow windows."""
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)

    if requested_max_seq_length is not None:
        if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 100_000 and requested_max_seq_length > tokenizer_limit:
            LOGGER.warning(
                "Requested max_seq_length=%d exceeds tokenizer.model_max_length=%d. Using %d instead.",
                requested_max_seq_length,
                tokenizer_limit,
                tokenizer_limit,
            )
            return tokenizer_limit
        return requested_max_seq_length

    candidates: list[int] = []
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 100_000:
        candidates.append(tokenizer_limit)

    position_limit = getattr(config, "max_position_embeddings", None)
    if isinstance(position_limit, int) and 0 < position_limit < 100_000:
        candidates.append(position_limit)

    if candidates:
        return min(candidates)

    LOGGER.warning(
        "Could not infer a reasonable model max length. Falling back to %d.",
        REASONABLE_MAX_LENGTH_FALLBACK,
    )
    return REASONABLE_MAX_LENGTH_FALLBACK
