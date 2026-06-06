#!/usr/bin/env python3
"""
Download CellExLink model checkpoints from Hugging Face.

This script keeps large model artifacts outside the GitHub repository while
making setup reproducible for users and SoftwareX reviewers.

Default checkpoints:
    NER: almire/CellExLink-bioformer16L
    NEN: almire/CellExLink-Sapbert

Examples
--------
Download both default checkpoints:

    python scripts/download_models.py --output-dir models

Download to explicit local paths:

    python scripts/download_models.py \
      --ner-model almire/CellExLink-bioformer16L \
      --nen-model almire/CellExLink-Sapbert \
      --output-dir models

Use a private Hugging Face token:

    HF_TOKEN=... python scripts/download_models.py --output-dir models
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

DEFAULT_NER_MODEL = "almire/CellExLink-bioformer16L"
DEFAULT_NEN_MODEL = "almire/CellExLink-Sapbert"
DEFAULT_OUTPUT_DIR = "models"
DEFAULT_MANIFEST = "models_manifest.json"


@dataclass(slots=True)
class DownloadedModel:
    role: str
    repo_id: str
    revision: Optional[str]
    local_dir: str
    file_count: int
    total_bytes: int
    digest_sha256: str


@dataclass(slots=True)
class ModelDownloadManifest:
    created_at_utc: str
    python_version: str
    platform: str
    huggingface_hub_version: Optional[str]
    output_dir: str
    models: list[DownloadedModel]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download CellExLink Hugging Face checkpoints."
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where model folders will be created. Default: %(default)s",
    )
    parser.add_argument(
        "--ner-model",
        default=DEFAULT_NER_MODEL,
        help="NER model repo ID or model path on Hugging Face Hub. Default: %(default)s",
    )
    parser.add_argument(
        "--nen-model",
        default=DEFAULT_NEN_MODEL,
        help="NEN/linker model repo ID or model path on Hugging Face Hub. Default: %(default)s",
    )
    parser.add_argument(
        "--ner-revision",
        default=None,
        help="Optional Hugging Face revision/commit/tag for the NER model.",
    )
    parser.add_argument(
        "--nen-revision",
        default=None,
        help="Optional Hugging Face revision/commit/tag for the NEN model.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help=(
            "Optional Hugging Face token. If omitted, HF_TOKEN or "
            "HUGGINGFACE_HUB_TOKEN is used when present."
        ),
    )
    parser.add_argument(
        "--skip-ner",
        action="store_true",
        help="Do not download the NER checkpoint.",
    )
    parser.add_argument(
        "--skip-nen",
        action="store_true",
        help="Do not download the NEN/linker checkpoint.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if files already exist in the Hugging Face cache.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only files already available in the local Hugging Face cache.",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=None,
        help=(
            "Optional file glob pattern to include. Can be repeated. "
            "Usually leave unset to download the full checkpoint."
        ),
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        default=None,
        help="Optional file glob pattern to ignore. Can be repeated.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help=(
            "Manifest JSON path. Default: <output-dir>/models_manifest.json. "
            "Use --no-manifest to disable."
        ),
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Do not write a manifest JSON file.",
    )
    return parser.parse_args(argv)


def get_hf_token(cli_token: Optional[str]) -> Optional[str]:
    return cli_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def safe_model_dir_name(repo_id: str) -> str:
    """Convert a Hub repo ID into a filesystem-friendly model directory name."""
    cleaned = repo_id.strip().rstrip("/")
    if not cleaned:
        raise ValueError("Model repo ID cannot be empty.")
    return cleaned.split("/")[-1].replace(" ", "_")


def iter_model_files(model_dir: Path) -> list[Path]:
    if not model_dir.exists():
        return []
    return sorted(p for p in model_dir.rglob("*") if p.is_file())


def hash_directory(model_dir: Path) -> tuple[int, int, str]:
    """
    Return file count, total bytes and a deterministic directory digest.

    The digest is based on relative file paths plus file content hashes. It is
    useful for detecting accidental changes to a downloaded checkpoint folder.
    """
    files = iter_model_files(model_dir)
    total_bytes = 0
    digest = hashlib.sha256()

    for path in files:
        rel = path.relative_to(model_dir).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        file_hash = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                total_bytes += len(chunk)
                file_hash.update(chunk)
        digest.update(file_hash.hexdigest().encode("ascii"))
        digest.update(b"\n")

    return len(files), total_bytes, digest.hexdigest()


def download_one_model(
    *,
    role: str,
    repo_id: str,
    output_dir: Path,
    revision: Optional[str],
    cache_dir: Optional[str],
    token: Optional[str],
    force_download: bool,
    local_files_only: bool,
    allow_patterns: Optional[list[str]],
    ignore_patterns: Optional[list[str]],
) -> DownloadedModel:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required. Install it with: "
            "pip install huggingface-hub"
        ) from exc

    target_dir = output_dir / safe_model_dir_name(repo_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "local_dir": str(target_dir),
        "revision": revision,
        "cache_dir": cache_dir,
        "token": token,
        "force_download": force_download,
        "local_files_only": local_files_only,
    }
    if allow_patterns:
        kwargs["allow_patterns"] = allow_patterns
    if ignore_patterns:
        kwargs["ignore_patterns"] = ignore_patterns

    # Remove None values to support older huggingface_hub versions.
    kwargs = {key: value for key, value in kwargs.items() if value is not None}

    print(f"[{role}] downloading {repo_id} -> {target_dir}")
    snapshot_path = Path(snapshot_download(**kwargs))

    # snapshot_download returns the local_dir path when local_dir is supplied in
    # current huggingface_hub versions, but older versions may return a cache
    # snapshot path. We report/check the user-facing target directory.
    if not target_dir.exists() and snapshot_path.exists():
        target_dir = snapshot_path

    file_count, total_bytes, digest = hash_directory(target_dir)
    if file_count == 0:
        raise RuntimeError(f"No files were downloaded for {repo_id} into {target_dir}")

    print(
        f"[{role}] downloaded {file_count} files "
        f"({format_bytes(total_bytes)}) to {target_dir}"
    )

    return DownloadedModel(
        role=role,
        repo_id=repo_id,
        revision=revision,
        local_dir=str(target_dir),
        file_count=file_count,
        total_bytes=total_bytes,
        digest_sha256=digest,
    )


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def get_huggingface_hub_version() -> Optional[str]:
    try:
        from importlib.metadata import version

        return version("huggingface_hub")
    except Exception:  # pragma: no cover
        return None


def write_manifest(manifest: ModelDownloadManifest, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(manifest)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if args.skip_ner and args.skip_nen:
        raise SystemExit("Nothing to download: both --skip-ner and --skip-nen were set.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    token = get_hf_token(args.token)

    downloaded: list[DownloadedModel] = []

    if not args.skip_ner:
        downloaded.append(
            download_one_model(
                role="ner",
                repo_id=args.ner_model,
                output_dir=output_dir,
                revision=args.ner_revision,
                cache_dir=args.cache_dir,
                token=token,
                force_download=args.force_download,
                local_files_only=args.local_files_only,
                allow_patterns=args.allow_pattern,
                ignore_patterns=args.ignore_pattern,
            )
        )

    if not args.skip_nen:
        downloaded.append(
            download_one_model(
                role="nen",
                repo_id=args.nen_model,
                output_dir=output_dir,
                revision=args.nen_revision,
                cache_dir=args.cache_dir,
                token=token,
                force_download=args.force_download,
                local_files_only=args.local_files_only,
                allow_patterns=args.allow_pattern,
                ignore_patterns=args.ignore_pattern,
            )
        )

    if not args.no_manifest:
        manifest_path = Path(args.manifest) if args.manifest else output_dir / DEFAULT_MANIFEST
        manifest = ModelDownloadManifest(
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            python_version=sys.version.replace("\n", " "),
            platform=platform.platform(),
            huggingface_hub_version=get_huggingface_hub_version(),
            output_dir=str(output_dir),
            models=downloaded,
        )
        write_manifest(manifest, manifest_path)
        print(f"Manifest written to: {manifest_path}")

    print("Model download complete.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
