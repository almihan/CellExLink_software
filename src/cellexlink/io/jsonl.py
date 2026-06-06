"""JSON, JSONL, and small tabular IO helpers for CellExLink."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional

from .schemas import PassageRecord, PathLike, coerce_passage_record, to_jsonable


def ensure_parent_dir(path: PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: PathLike, *, skip_blank: bool = True) -> Iterator[dict[str, Any]]:
    """Yield dictionaries from a JSONL file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line and skip_blank:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object on line {line_number} of {path}.")
            yield value


def load_jsonl(path: PathLike, *, skip_blank: bool = True) -> list[dict[str, Any]]:
    return list(read_jsonl(path, skip_blank=skip_blank))


def write_jsonl(
    records: Iterable[Any],
    path: PathLike,
    *,
    append: bool = False,
    sort_keys: bool = False,
) -> Path:
    """Write an iterable of records to JSONL."""
    path = ensure_parent_dir(path)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    to_jsonable(record),
                    ensure_ascii=False,
                    sort_keys=sort_keys,
                )
                + "\n"
            )
    return path


def atomic_write_jsonl(
    records: Iterable[Any],
    path: PathLike,
    *,
    sort_keys: bool = False,
) -> Path:
    """Write JSONL through a temporary file and atomically replace the target."""
    path = ensure_parent_dir(path)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        delete=False,
        suffix=".tmp",
    ) as handle:
        tmp_path = Path(handle.name)
        for record in records:
            handle.write(
                json.dumps(
                    to_jsonable(record),
                    ensure_ascii=False,
                    sort_keys=sort_keys,
                )
                + "\n"
            )
    tmp_path.replace(path)
    return path


def load_json(path: PathLike) -> Any:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(
    value: Any,
    path: PathLike,
    *,
    indent: Optional[int] = 2,
    sort_keys: bool = False,
) -> Path:
    path = ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            to_jsonable(value),
            handle,
            ensure_ascii=False,
            indent=indent,
            sort_keys=sort_keys,
        )
        handle.write("\n")
    return path


def read_passage_records(path: PathLike) -> Iterator[PassageRecord]:
    """Yield :class:`PassageRecord` objects from CellExLink JSONL."""
    for row in read_jsonl(path):
        yield coerce_passage_record(row)


def load_passage_records(path: PathLike) -> list[PassageRecord]:
    return list(read_passage_records(path))


def write_passage_records(records: Iterable[PassageRecord | Mapping[str, Any]], path: PathLike) -> Path:
    """Write CellExLink passage records using the package JSONL schema."""
    return write_jsonl((coerce_passage_record(record) for record in records), path)


def read_tsv(path: PathLike, *, has_header: bool = True) -> list[dict[str, str]]:
    """Read a small UTF-8 TSV file.

    This helper is intentionally simple. Use pandas for large analysis tables.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"TSV file does not exist: {path}")

    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline()
        if not first:
            return []
        first_values = first.rstrip("\n").split("\t")
        if has_header:
            header = first_values
        else:
            header = [f"column_{index}" for index in range(len(first_values))]
            rows.append(dict(zip(header, first_values)))

        for line_number, line in enumerate(handle, start=2):
            values = line.rstrip("\n").split("\t")
            if len(values) != len(header):
                raise ValueError(
                    f"Expected {len(header)} columns on line {line_number} of {path}, "
                    f"found {len(values)}."
                )
            rows.append(dict(zip(header, values)))
    return rows


def write_tsv(rows: Iterable[Mapping[str, Any]], path: PathLike, *, header: Optional[list[str]] = None) -> Path:
    """Write a small UTF-8 TSV file from mappings."""
    rows = list(rows)
    path = ensure_parent_dir(path)

    if header is None:
        header = []
        for row in rows:
            for key in row.keys():
                if key not in header:
                    header.append(str(key))

    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            handle.write("\t".join(_tsv_cell(row.get(key, "")) for key in header) + "\n")
    return path


def _tsv_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("\t", " ").replace("\n", " ").replace("\r", " ")
