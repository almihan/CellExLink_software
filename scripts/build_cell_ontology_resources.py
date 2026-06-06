#!/usr/bin/env python3
"""
Build a CellExLink Cell Ontology JSONL resource.

The output schema is accepted by ``cellexlink.normalization.ontology``:

    {
      "norm_concept_id": "CL:0000077",
      "norm_preferred_label": "mesothelial cell",
      "synonyms": ["mesotheliocyte", ...],
      "namespace": "cell",
      "alt_ids": [...],
      "definition": "...",
      "parents": [...]
    }

The script can parse a local OBO file, download the public CL OBO file, and
merge additional curated aliases from TSV/CSV/JSONL files. This lets you keep a
reproducible packaged resource without committing heavyweight ontology tooling
or large external downloads to the repository.

Examples
--------
Build from the public CL OBO PURL:

    python scripts/build_cell_ontology_resource.py \
      --output-jsonl src/cellexlink/resources/cell_ontology_v2025-12-17.jsonl

Build from a local OBO file and a curated alias TSV:

    python scripts/build_cell_ontology_resource.py \
      --input-obo data/raw/cl.obo \
      --extra-aliases data/raw/celllink_training_aliases.tsv \
      --output-jsonl src/cellexlink/resources/cell_ontology_v2025-12-17.jsonl
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

DEFAULT_CL_OBO_URL = "https://purl.obolibrary.org/obo/cl.obo"
DEFAULT_RELEASE_LABEL = "2025-12-17"
DEFAULT_OUTPUT_TEMPLATE = "src/cellexlink/resources/cell_ontology_v{release_label}.jsonl"
USER_AGENT = "CellExLink ontology resource builder/0.1"
SYNONYM_RE = re.compile(r'^"(?P<text>(?:[^"\\]|\\.)*)"\s*(?P<scope>EXACT|BROAD|NARROW|RELATED)?')
DEF_RE = re.compile(r'^"(?P<text>(?:[^"\\]|\\.)*)"')
CL_ID_RE = re.compile(r"CL:\d+")


@dataclass(slots=True)
class OntologyTerm:
    identifier: str
    name: str
    namespace: str = ""
    synonyms: set[str] = field(default_factory=set)
    alt_ids: set[str] = field(default_factory=set)
    definition: str = ""
    parents: set[str] = field(default_factory=set)
    is_obsolete: bool = False

    def to_json_record(self) -> dict[str, Any]:
        return {
            "norm_concept_id": self.identifier,
            "norm_preferred_label": self.name,
            "synonyms": sorted(s for s in self.synonyms if s and s != self.name),
            "namespace": self.namespace,
            "alt_ids": sorted(self.alt_ids),
            "definition": self.definition,
            "parents": sorted(self.parents),
        }


@dataclass(slots=True)
class BuildManifest:
    created_at_utc: str
    python_version: str
    platform: str
    source: str
    source_sha256: str
    output_jsonl: str
    release_label: str
    include_obsolete: bool
    terms_written: int
    aliases_added: int


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the CellExLink Cell Ontology JSONL resource."
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--input-obo",
        default=None,
        help="Local Cell Ontology OBO file. If omitted, --download-url is used.",
    )
    source_group.add_argument(
        "--input-jsonl",
        default=None,
        help=(
            "Existing ontology JSONL to normalize into the CellExLink schema. "
            "Useful for BioPortal or curated exports."
        ),
    )
    source_group.add_argument(
        "--input-tsv",
        default=None,
        help=(
            "Existing ontology TSV/CSV to normalize into the CellExLink schema. "
            "Expected columns include id/concept_id/cl_id and name/label."
        ),
    )
    parser.add_argument(
        "--download-url",
        default=DEFAULT_CL_OBO_URL,
        help="OBO URL used when --input-obo/--input-jsonl/--input-tsv is omitted.",
    )
    parser.add_argument(
        "--release-label",
        default=DEFAULT_RELEASE_LABEL,
        help=(
            "Label embedded in the default output filename. For strict "
            "reproducibility, set this to the ontology release date you used."
        ),
    )
    parser.add_argument(
        "--output-jsonl",
        default=None,
        help=(
            "Output JSONL path. Default: "
            "src/cellexlink/resources/cell_ontology_v<release-label>.jsonl"
        ),
    )
    parser.add_argument(
        "--extra-aliases",
        action="append",
        default=None,
        help=(
            "Optional TSV/CSV/JSONL alias file to merge. Can be repeated. "
            "Expected fields: cl_id/concept_id/id and alias/name/synonym."
        ),
    )
    parser.add_argument(
        "--include-obsolete",
        action="store_true",
        help="Include obsolete ontology terms. Default: exclude obsolete terms.",
    )
    parser.add_argument(
        "--keep-non-cl",
        action="store_true",
        help="Keep non-CL identifiers. Default: keep only CL:* terms.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP timeout in seconds for ontology download. Default: %(default)s",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help=(
            "Build manifest path. Default: <output-jsonl>.manifest.json. "
            "Use --no-manifest to disable."
        ),
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Do not write a build manifest.",
    )
    parser.add_argument(
        "--validate-with-package",
        action="store_true",
        help=(
            "After writing, import cellexlink.normalization.ontology and check "
            "that the resource can be loaded."
        ),
    )
    return parser.parse_args(argv)


def unescape_obo_string(value: str) -> str:
    return value.replace('\\"', '"').replace('\\n', '\n').strip()


def extract_quoted_text(raw_value: str, pattern: re.Pattern[str]) -> str:
    match = pattern.match(raw_value.strip())
    if not match:
        return ""
    return unescape_obo_string(match.group("text"))


def parse_obo(path: str | Path, *, keep_non_cl: bool = False) -> dict[str, OntologyTerm]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"OBO file does not exist: {path}")

    terms: dict[str, OntologyTerm] = {}
    current: Optional[dict[str, list[str]]] = None

    def flush_current() -> None:
        nonlocal current
        if not current:
            return
        identifier = first_value(current, "id")
        name = first_value(current, "name")
        if not identifier or not name:
            current = None
            return
        if not keep_non_cl and not identifier.startswith("CL:"):
            current = None
            return

        term = OntologyTerm(
            identifier=identifier,
            name=name,
            namespace=first_value(current, "namespace") or "",
            definition=extract_quoted_text(first_value(current, "def") or "", DEF_RE),
            is_obsolete=(first_value(current, "is_obsolete") or "").lower() == "true",
        )
        for raw_synonym in current.get("synonym", []):
            synonym = extract_quoted_text(raw_synonym, SYNONYM_RE)
            if synonym:
                term.synonyms.add(synonym)
        for raw_alt in current.get("alt_id", []):
            raw_alt = raw_alt.strip()
            if raw_alt:
                term.alt_ids.add(raw_alt)
        for raw_parent in current.get("is_a", []):
            match = CL_ID_RE.search(raw_parent)
            if match:
                term.parents.add(match.group(0))
        for raw_relationship in current.get("relationship", []):
            # Relationship lines often contain a target CL ID, for example:
            # relationship: part_of CL:0000000 ! cell
            match = CL_ID_RE.search(raw_relationship)
            if match:
                term.parents.add(match.group(0))

        terms[identifier] = term
        current = None

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line or line.startswith("!"):
                continue
            if line == "[Term]":
                flush_current()
                current = {}
                continue
            if line.startswith("["):
                flush_current()
                current = None
                continue
            if current is None or ":" not in line:
                continue
            tag, value = line.split(":", 1)
            current.setdefault(tag.strip(), []).append(value.strip())
    flush_current()

    return terms


def first_value(mapping: dict[str, list[str]], key: str) -> Optional[str]:
    values = mapping.get(key)
    if not values:
        return None
    value = values[0].strip()
    return value or None


def parse_existing_jsonl(path: str | Path, *, keep_non_cl: bool = False) -> dict[str, OntologyTerm]:
    path = Path(path)
    terms: dict[str, OntologyTerm] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            identifier = first_nonempty(record, ["norm_concept_id", "concept_id", "identifier", "id", "cl_id", "obo_id"])
            name = first_nonempty(record, ["norm_preferred_label", "preferred_label", "label", "name", "term"])
            if not identifier or not name:
                raise ValueError(f"Missing identifier/name on line {line_no} of {path}")
            if not keep_non_cl and not identifier.startswith("CL:"):
                continue
            term = OntologyTerm(
                identifier=identifier,
                name=name,
                namespace=str(record.get("namespace", "") or ""),
                definition=str(record.get("definition", "") or record.get("def", "") or ""),
                is_obsolete=bool(record.get("is_obsolete", False)),
            )
            for key in ["synonyms", "aliases", "names", "exact_synonyms", "related_synonyms"]:
                for value in as_list(record.get(key)):
                    if value != name:
                        term.synonyms.add(value)
            for value in as_list(record.get("alt_ids") or record.get("alt_id")):
                term.alt_ids.add(value)
            for value in as_list(record.get("parents") or record.get("is_a")):
                term.parents.add(value)
            terms[identifier] = term
    return terms


def parse_existing_table(path: str | Path, *, keep_non_cl: bool = False) -> dict[str, OntologyTerm]:
    path = Path(path)
    delimiter = "," if path.suffix.lower() == ".csv" else "\t"
    terms: dict[str, OntologyTerm] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"Table has no header: {path}")
        for row_no, row in enumerate(reader, start=2):
            identifier = first_nonempty(row, ["norm_concept_id", "concept_id", "identifier", "id", "cl_id", "obo_id"])
            name = first_nonempty(row, ["norm_preferred_label", "preferred_label", "label", "name", "term"])
            if not identifier or not name:
                raise ValueError(f"Missing identifier/name on row {row_no} of {path}")
            if not keep_non_cl and not identifier.startswith("CL:"):
                continue
            term = terms.setdefault(
                identifier,
                OntologyTerm(
                    identifier=identifier,
                    name=name,
                    namespace=str(row.get("namespace", "") or ""),
                    definition=str(row.get("definition", "") or row.get("def", "") or ""),
                ),
            )
            alias = first_nonempty(row, ["alias", "synonym", "name", "label", "term"])
            if alias and alias != term.name:
                term.synonyms.add(alias)
    return terms


def first_nonempty(record: dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    lower_map = {str(key).lower(): value for key, value in record.items()}
    for key in keys:
        value = record.get(key)
        if value is None:
            value = lower_map.get(key.lower())
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        # Accept simple pipe-separated or semicolon-separated alias fields.
        if "|" in value:
            pieces = value.split("|")
        elif ";" in value:
            pieces = value.split(";")
        else:
            pieces = [value]
        return [piece.strip() for piece in pieces if piece.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def download_obo(url: str, *, timeout: int) -> Path:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
    except urllib.error.HTTPError as exc:  # pragma: no cover
        raise RuntimeError(f"HTTP {exc.code} while downloading {url}: {exc.reason}") from exc
    except urllib.error.URLError as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to download {url}: {exc.reason}") from exc

    tmp = tempfile.NamedTemporaryFile(prefix="cellexlink_cl_", suffix=".obo", delete=False)
    with tmp:
        tmp.write(data)
    return Path(tmp.name)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_extra_aliases(terms: dict[str, OntologyTerm], paths: Iterable[str | Path]) -> int:
    aliases_added = 0
    for path in paths:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Extra alias file does not exist: {path}")
        if path.suffix.lower() == ".jsonl":
            aliases_added += merge_extra_aliases_jsonl(terms, path)
        else:
            aliases_added += merge_extra_aliases_table(terms, path)
    return aliases_added


def merge_extra_aliases_jsonl(terms: dict[str, OntologyTerm], path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            identifier = first_nonempty(record, ["cl_id", "concept_id", "norm_concept_id", "identifier", "id"])
            aliases = []
            for key in ["alias", "synonym", "name", "mention", "text", "aliases", "synonyms", "names"]:
                aliases.extend(as_list(record.get(key)))
            if not identifier or not aliases:
                raise ValueError(f"Missing concept ID or alias on line {line_no} of {path}")
            if identifier not in terms:
                continue
            for alias in aliases:
                if alias and alias != terms[identifier].name and alias not in terms[identifier].synonyms:
                    terms[identifier].synonyms.add(alias)
                    count += 1
    return count


def merge_extra_aliases_table(terms: dict[str, OntologyTerm], path: Path) -> int:
    delimiter = "," if path.suffix.lower() == ".csv" else "\t"
    count = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"Alias table has no header: {path}")
        for row_no, row in enumerate(reader, start=2):
            identifier = first_nonempty(row, ["cl_id", "concept_id", "norm_concept_id", "identifier", "id"])
            alias = first_nonempty(row, ["alias", "synonym", "name", "mention", "text"])
            if not identifier or not alias:
                raise ValueError(f"Missing concept ID or alias on row {row_no} of {path}")
            if identifier not in terms:
                continue
            if alias != terms[identifier].name and alias not in terms[identifier].synonyms:
                terms[identifier].synonyms.add(alias)
                count += 1
    return count


def write_resource(
    terms: dict[str, OntologyTerm],
    output_jsonl: str | Path,
    *,
    include_obsolete: bool = False,
) -> int:
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for identifier in sorted(terms):
            term = terms[identifier]
            if term.is_obsolete and not include_obsolete:
                continue
            record = term.to_json_record()
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            written += 1
    return written


def validate_resource_with_package(output_jsonl: Path) -> None:
    try:
        from cellexlink.normalization.ontology import load_cell_ontology_terms
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Could not import cellexlink.normalization.ontology. "
            "Install the package with `pip install -e .` before using "
            "--validate-with-package."
        ) from exc
    term_entries, concept_metadata = load_cell_ontology_terms(output_jsonl)
    print(
        f"Validated resource with package loader: "
        f"{len(term_entries)} aliases, {len(concept_metadata)} concepts."
    )


def write_manifest(manifest: BuildManifest, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    return path


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    output_jsonl = Path(
        args.output_jsonl
        or DEFAULT_OUTPUT_TEMPLATE.format(release_label=args.release_label)
    )

    source_description: str
    source_path: Path
    cleanup_source = False

    if args.input_jsonl:
        source_path = Path(args.input_jsonl)
        source_description = str(source_path)
        terms = parse_existing_jsonl(source_path, keep_non_cl=args.keep_non_cl)
    elif args.input_tsv:
        source_path = Path(args.input_tsv)
        source_description = str(source_path)
        terms = parse_existing_table(source_path, keep_non_cl=args.keep_non_cl)
    else:
        if args.input_obo:
            source_path = Path(args.input_obo)
            source_description = str(source_path)
        else:
            print(f"Downloading Cell Ontology OBO from: {args.download_url}")
            source_path = download_obo(args.download_url, timeout=args.timeout)
            source_description = args.download_url
            cleanup_source = True
        terms = parse_obo(source_path, keep_non_cl=args.keep_non_cl)

    if not terms:
        raise SystemExit("No ontology terms were parsed from the selected source.")

    aliases_added = 0
    if args.extra_aliases:
        aliases_added = merge_extra_aliases(terms, args.extra_aliases)
        print(f"Merged {aliases_added} extra aliases from curated files.")

    terms_written = write_resource(
        terms,
        output_jsonl,
        include_obsolete=args.include_obsolete,
    )
    print(f"Wrote {terms_written} Cell Ontology concepts to: {output_jsonl}")

    if args.validate_with_package:
        validate_resource_with_package(output_jsonl)

    if not args.no_manifest:
        manifest_path = Path(args.manifest) if args.manifest else output_jsonl.with_suffix(output_jsonl.suffix + ".manifest.json")
        manifest = BuildManifest(
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            python_version=sys.version.replace("\n", " "),
            platform=platform.platform(),
            source=source_description,
            source_sha256=file_sha256(source_path),
            output_jsonl=str(output_jsonl),
            release_label=args.release_label,
            include_obsolete=args.include_obsolete,
            terms_written=terms_written,
            aliases_added=aliases_added,
        )
        write_manifest(manifest, manifest_path)
        print(f"Manifest written to: {manifest_path}")

    if cleanup_source:
        try:
            source_path.unlink()
        except OSError:
            pass

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
