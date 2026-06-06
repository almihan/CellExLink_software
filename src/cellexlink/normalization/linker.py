"""
Cell Ontology normalization/linking for CellExLink.

This module turns recognized cell-type mentions into Cell Ontology identifiers.
It is the package-style replacement for the old script-centered normalization
code. It keeps the same main user-facing behavior:

    normalize_bioc(input_xml, output_xml, model_path=..., cell_types=...)

The implementation has three stages:
1. Load Cell Ontology labels/synonyms.
2. Retrieve candidate concepts with a biomedical encoder.
3. Re-rank candidates with lexical and abbreviation-aware features.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence
from xml.etree import ElementTree as ET

import numpy as np

from .abbreviations import (
    AbbreviationLookup,
    abbreviation_threshold_for,
    build_document_abbreviation_lookup,
    find_ab3p_long_form_with_fallback,
    is_abbreviation_like,
    load_abbreviation_identifier_lookup,
    normalize_abbreviation_key,
)
from .ontology import (
    ConceptMetadata,
    TermEntry,
    default_ontology_path,
    load_cell_ontology_terms,
)
from .stemmer import normalize_text

DEFAULT_TOPN = 1
AMBIGUOUS_TOPN = 5
DEFAULT_NEN_MODEL = "almire/CellExLink-Sapbert"
DEFAULT_ABBREVIATIONS_FILENAME = "abbreviations.tsv"
EL_RUNTIME_SUMMARY_FILENAME = "el_predict_runtime_summary.json"


@dataclass(frozen=True, slots=True)
class MentionRecord:
    document_key: str
    document_id: str
    passage_index: int
    annotation_index: int
    mention_text: str
    normalized_text: str


@dataclass(slots=True)
class NormalizationCandidate:
    identifier: str
    name: str
    preferred_label: str
    embedding_score: float
    final_score: float
    source: str = "model_normal"
    is_preferred: bool = False
    matched_alias: Optional[str] = None
    exact_synonym_match: float = 0.0
    token_overlap: float = 0.0
    preferred_overlap: float = 0.0
    parenthetical_match: float = 0.0
    sequence_ratio: float = 0.0
    abbreviation_method: Optional[str] = None
    expanded_long_form: Optional[str] = None
    ab3p_method: Optional[str] = None
    ab3p_matched_key: Optional[str] = None
    ab3p_match_score: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class NormalizationResult:
    mention_text: str
    normalized_text: str
    document_key: Optional[str]
    candidates: list[NormalizationCandidate] = field(default_factory=list)

    @property
    def best(self) -> Optional[NormalizationCandidate]:
        return self.candidates[0] if self.candidates else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mention_text": self.mention_text,
            "normalized_text": self.normalized_text,
            "document_key": self.document_key,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(slots=True)
class EncoderHandle:
    kind: str
    model: Any
    tokenizer: Any = None
    device: Optional[str] = None


@dataclass(slots=True)
class RuntimeSummary:
    elapsed_seconds: float
    total_unique_mentions: int
    total_annotations: int
    model_name: str
    stats: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.total_unique_mentions and self.elapsed_seconds > 0:
            data["mentions_per_second"] = self.total_unique_mentions / self.elapsed_seconds
            data["ms_per_mention"] = (self.elapsed_seconds * 1000.0) / self.total_unique_mentions
        return data


def default_abbreviations_path() -> Path:
    candidate = resources.files("cellexlink").joinpath(
        "resources", DEFAULT_ABBREVIATIONS_FILENAME
    )
    return Path(str(candidate))


def resolve_model_reference(model_reference: str | Path) -> tuple[str, str]:
    """Return model reference string and a readable model label for BioC infon keys."""
    model_reference_str = str(model_reference)
    model_reference_path = Path(model_reference_str)
    if model_reference_path.exists():
        resolved_path = model_reference_path.resolve()
        return str(resolved_path), resolved_path.name
    return model_reference_str, Path(model_reference_str).name


def _clean_model_label(model_label: str) -> str:
    label = Path(str(model_label)).name or str(model_label)
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    return label or "CellExLink_NEN"


def is_sentence_transformers_model(model_name_or_path: str | Path) -> bool:
    model_path = Path(str(model_name_or_path))
    return model_path.is_dir() and (model_path / "modules.json").is_file()


def load_encoder(
    model_name_or_path: str | Path,
    *,
    device: Optional[str] = None,
    prefer_sentence_transformers: bool = True,
    trust_remote_code: bool = False,
) -> EncoderHandle:
    """Load either a sentence-transformers encoder or a Hugging Face encoder."""
    model_name_or_path = str(model_name_or_path)

    if device is None:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # pragma: no cover - torch missing at import time
            device = "cpu"

    if prefer_sentence_transformers:
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name_or_path, device=device)
            return EncoderHandle(kind="sentence_transformers", model=model, device=device)
        except Exception:
            # Some checkpoints are plain transformers checkpoints. Fall back.
            pass

    from transformers import AutoModel, AutoTokenizer

    import torch

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModel.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    model = model.to(torch.device(device))
    model.eval()
    return EncoderHandle(
        kind="transformers",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )


def encode_texts(
    encoder: EncoderHandle,
    texts: Sequence[str],
    *,
    batch_size: int = 128,
    max_length: int = 32,
) -> np.ndarray:
    """Encode names or query mentions as a NumPy array."""
    texts = [str(text) for text in texts]
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    if encoder.kind == "sentence_transformers":
        embeddings = encoder.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    import torch

    all_reps: list[np.ndarray] = []
    tokenizer = encoder.tokenizer
    model = encoder.model
    device = encoder.device or "cpu"

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            toks = tokenizer.batch_encode_plus(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            toks = {key: value.to(device) for key, value in toks.items()}
            output = model(**toks)
            # Keep compatibility with the original code, which used CLS vectors.
            cls_rep = output.last_hidden_state[:, 0, :]
            all_reps.append(cls_rep.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(all_reps, axis=0)


def cosine_similarity_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Cosine similarity without requiring scipy."""
    if left.size == 0 or right.size == 0:
        return np.zeros((left.shape[0], right.shape[0]), dtype=np.float32)

    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True)
    left_norm[left_norm == 0] = 1.0
    right_norm[right_norm == 0] = 1.0
    return (left / left_norm) @ (right / right_norm).T


def token_jaccard(left: str, right: str) -> float:
    left_tokens = set(normalize_text(left).split())
    right_tokens = set(normalize_text(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def sequence_ratio(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio()


def has_parenthetical_relation(left: str, right: str) -> float:
    left = str(left)
    right = str(right)
    if "(" in left and ")" in left:
        inside = re.findall(r"\(([^()]*)\)", left)
        if any(normalize_text(item) == normalize_text(right) for item in inside):
            return 1.0
    if "(" in right and ")" in right:
        inside = re.findall(r"\(([^()]*)\)", right)
        if any(normalize_text(item) == normalize_text(left) for item in inside):
            return 1.0
    return 0.0


class CellOntologyLinker:
    """Reusable Cell Ontology linker."""

    def __init__(
        self,
        *,
        model_path: str | Path = DEFAULT_NEN_MODEL,
        term_entries: list[TermEntry],
        concept_metadata: dict[str, ConceptMetadata],
        abbreviation_lookup: Optional[AbbreviationLookup] = None,
        document_abbreviation_lookup: Optional[dict[str, dict[str, str]]] = None,
        batch_size: int = 128,
        topn: int = DEFAULT_TOPN,
        ambiguous_topn: int = AMBIGUOUS_TOPN,
        device: Optional[str] = None,
        trust_remote_code: bool = False,
        verbose: bool = True,
    ) -> None:
        self.model_path = str(model_path)
        self.term_entries = term_entries
        self.concept_metadata = concept_metadata
        self.abbreviation_lookup = abbreviation_lookup or AbbreviationLookup()
        self.document_abbreviation_lookup = document_abbreviation_lookup or {}
        self.batch_size = batch_size
        self.topn = topn
        self.ambiguous_topn = ambiguous_topn
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.verbose = verbose

        self.encoder: Optional[EncoderHandle] = None
        self.dictionary_embeddings: Optional[np.ndarray] = None
        self.abbreviation_embeddings: Optional[np.ndarray] = None
        self._query_cache: dict[tuple[str, str], list[NormalizationCandidate]] = {}
        self._abbreviation_result_cache: dict[str, Optional[dict[str, Any]]] = {}
        self._mention_embedding_cache: dict[str, np.ndarray] = {}

    @classmethod
    def from_files(
        cls,
        *,
        ontology_path: str | Path,
        model_path: str | Path = DEFAULT_NEN_MODEL,
        abbreviations_path: str | Path | Iterable[str | Path] | None = None,
        disable_abbreviations: bool = False,
        document_abbreviation_lookup: Optional[dict[str, dict[str, str]]] = None,
        batch_size: int = 128,
        topn: int = DEFAULT_TOPN,
        ambiguous_topn: int = AMBIGUOUS_TOPN,
        device: Optional[str] = None,
        trust_remote_code: bool = False,
        verbose: bool = True,
    ) -> "CellOntologyLinker":
        term_entries, concept_metadata = load_cell_ontology_terms(ontology_path)
        abbreviation_lookup = AbbreviationLookup()
        if not disable_abbreviations and abbreviations_path not in [None, "", "."]:
            abbreviation_lookup = load_abbreviation_identifier_lookup(
                abbreviations_path,
                verbose=verbose,
            )

        linker = cls(
            model_path=model_path,
            term_entries=term_entries,
            concept_metadata=concept_metadata,
            abbreviation_lookup=abbreviation_lookup,
            document_abbreviation_lookup=document_abbreviation_lookup,
            batch_size=batch_size,
            topn=topn,
            ambiguous_topn=ambiguous_topn,
            device=device,
            trust_remote_code=trust_remote_code,
            verbose=verbose,
        )
        linker.prepare()
        return linker

    def prepare(self) -> None:
        """Load encoder and pre-encode ontology aliases and abbreviation keys."""
        if self.encoder is None:
            if self.verbose:
                print(f"Loading normalization encoder: {self.model_path}")
            self.encoder = load_encoder(
                self.model_path,
                device=self.device,
                trust_remote_code=self.trust_remote_code,
            )

        if self.dictionary_embeddings is None:
            dictionary_names = [entry.name for entry in self.term_entries]
            if self.verbose:
                print(
                    "Encoding {} Cell Ontology labels/synonyms".format(
                        len(dictionary_names)
                    )
                )
            self.dictionary_embeddings = encode_texts(
                self.encoder,
                dictionary_names,
                batch_size=self.batch_size,
            )

        if (
            self.abbreviation_lookup
            and self.abbreviation_lookup.all_keys
            and self.abbreviation_embeddings is None
        ):
            if self.verbose:
                print(
                    "Encoding {} abbreviation keys".format(
                        len(self.abbreviation_lookup.all_keys)
                    )
                )
            self.abbreviation_embeddings = encode_texts(
                self.encoder,
                self.abbreviation_lookup.all_keys,
                batch_size=self.batch_size,
            )

    def link_mentions(
        self,
        mentions: Iterable[str | tuple[str, str]],
    ) -> dict[tuple[Optional[str], str], NormalizationResult]:
        """
        Link a collection of mentions.

        Each item can be either a mention string or ``(document_key, mention)``.
        """
        outputs: dict[tuple[Optional[str], str], NormalizationResult] = {}
        for item in mentions:
            if isinstance(item, tuple):
                document_key, mention_text = item
            else:
                document_key, mention_text = None, item
            normalized = normalize_text(mention_text)
            outputs[(document_key, normalized)] = self.link_mention(
                mention_text=mention_text,
                document_key=document_key,
            )
        return outputs

    def link_mention(
        self,
        mention_text: str,
        *,
        document_key: Optional[str] = None,
    ) -> NormalizationResult:
        """Link one mention to Cell Ontology candidates."""
        self.prepare()
        normalized_text = normalize_text(mention_text)
        stats_source = "model_normal"

        if is_abbreviation_like(mention_text) and self.abbreviation_lookup:
            abbreviation_result = self._resolve_abbreviation(
                mention_text=mention_text,
                document_key=document_key,
            )
            if abbreviation_result is not None:
                if abbreviation_result["type"] == "direct":
                    candidate = NormalizationCandidate(
                        identifier=abbreviation_result["identifier"],
                        name=abbreviation_result["short_form"],
                        preferred_label=self._preferred_label_for(
                            abbreviation_result["identifier"],
                            fallback=abbreviation_result["short_form"],
                        ),
                        embedding_score=float(abbreviation_result["score"]),
                        final_score=float(abbreviation_result["score"]),
                        source="abbreviation_direct",
                        matched_alias=abbreviation_result["short_form"],
                        abbreviation_method=abbreviation_result.get("method"),
                    )
                    return NormalizationResult(
                        mention_text=mention_text,
                        normalized_text=normalized_text,
                        document_key=document_key,
                        candidates=[candidate],
                    )

                if abbreviation_result["type"] == "ambiguous_long_form":
                    query_text = normalize_text(abbreviation_result["expanded_long_form"])
                    candidates = self._retrieve_and_rerank(
                        query_text=query_text,
                        topn=self.ambiguous_topn,
                        initial_k=50,
                        cache_prefix="ambiguous_long_form",
                    )
                    if candidates:
                        best = candidates[0]
                        best.source = "abbreviation_ambiguous_via_long_form"
                        best.abbreviation_method = abbreviation_result.get("method")
                        best.expanded_long_form = abbreviation_result.get("expanded_long_form")
                        best.ab3p_method = abbreviation_result.get("ab3p_method")
                        best.ab3p_matched_key = abbreviation_result.get("ab3p_matched_key")
                        best.ab3p_match_score = abbreviation_result.get("ab3p_match_score")
                        return NormalizationResult(
                            mention_text=mention_text,
                            normalized_text=normalized_text,
                            document_key=document_key,
                            candidates=[best],
                        )
                stats_source = abbreviation_result.get("type", stats_source)

        candidates = self._retrieve_and_rerank(
            query_text=normalized_text,
            topn=self.topn,
            initial_k=max(self.topn * 10, 20),
            cache_prefix="normal",
        )
        for candidate in candidates:
            candidate.source = "model_normal" if candidate.source == "model_normal" else candidate.source

        return NormalizationResult(
            mention_text=mention_text,
            normalized_text=normalized_text,
            document_key=document_key,
            candidates=candidates,
        )

    def _resolve_abbreviation(
        self,
        *,
        mention_text: str,
        document_key: Optional[str],
    ) -> Optional[dict[str, Any]]:
        match = self._find_best_abbreviation_key(mention_text)
        if match is None:
            return None

        matched_key = match["matched_key"]
        score = float(match["score"])
        method = str(match["method"])

        direct_match = self.abbreviation_lookup.direct_lookup.get(matched_key)
        if direct_match is not None:
            return {
                "type": "direct",
                "matched_key": matched_key,
                "score": score,
                "method": method,
                "short_form": direct_match[0],
                "identifier": direct_match[1],
            }

        candidates = self.abbreviation_lookup.ambiguous_candidates.get(matched_key, [])
        if not candidates:
            return None

        doc_lookup = self.document_abbreviation_lookup.get(str(document_key), {}) if document_key else {}
        expanded_long_form, ab3p_method, ab3p_matched_key, ab3p_score = (
            find_ab3p_long_form_with_fallback(doc_lookup, matched_key)
        )
        if not expanded_long_form:
            return {
                "type": "ambiguous_unresolved",
                "matched_key": matched_key,
                "score": score,
                "method": method,
            }

        return {
            "type": "ambiguous_long_form",
            "matched_key": matched_key,
            "score": score,
            "method": method,
            "expanded_long_form": expanded_long_form,
            "ab3p_method": ab3p_method,
            "ab3p_matched_key": ab3p_matched_key,
            "ab3p_match_score": ab3p_score,
        }

    def _find_best_abbreviation_key(self, mention_text: str) -> Optional[dict[str, Any]]:
        if not self.abbreviation_lookup or not self.abbreviation_lookup.all_keys:
            return None

        if mention_text in self._abbreviation_result_cache:
            return self._abbreviation_result_cache[mention_text]

        mention_key = normalize_abbreviation_key(mention_text)
        if not mention_key:
            self._abbreviation_result_cache[mention_text] = None
            return None

        threshold = abbreviation_threshold_for(mention_text)
        if threshold == 1.0:
            if mention_key in self.abbreviation_lookup.key_to_candidates:
                result = {
                    "matched_key": mention_key,
                    "score": 1.0,
                    "method": "exact_short_abbreviation",
                }
                self._abbreviation_result_cache[mention_text] = result
                return result
            self._abbreviation_result_cache[mention_text] = None
            return None

        if self.encoder is None or self.abbreviation_embeddings is None:
            self.prepare()
        assert self.encoder is not None
        assert self.abbreviation_embeddings is not None

        if mention_key in self._mention_embedding_cache:
            query_rep = self._mention_embedding_cache[mention_key]
        else:
            query_rep = encode_texts(
                self.encoder,
                [mention_key],
                batch_size=self.batch_size,
            )
            self._mention_embedding_cache[mention_key] = query_rep

        sims = cosine_similarity_matrix(query_rep, self.abbreviation_embeddings)
        best_idx = int(np.argmax(sims[0]))
        best_score = float(sims[0][best_idx])
        best_key = self.abbreviation_lookup.all_keys[best_idx]

        if best_score >= threshold:
            result = {
                "matched_key": best_key,
                "score": best_score,
                "method": "encoder_abbreviation_match",
            }
            self._abbreviation_result_cache[mention_text] = result
            return result

        self._abbreviation_result_cache[mention_text] = None
        return None

    def _retrieve_and_rerank(
        self,
        *,
        query_text: str,
        topn: int,
        initial_k: int,
        cache_prefix: str,
    ) -> list[NormalizationCandidate]:
        cache_key = (cache_prefix, query_text)
        if cache_key in self._query_cache:
            return [candidate for candidate in self._query_cache[cache_key]]

        retrieved = self._retrieve_candidates(
            query_text=query_text,
            topn=topn,
            initial_k=initial_k,
        )
        reranked = self._rerank_candidates(query_text, retrieved)
        self._query_cache[cache_key] = reranked
        return [candidate for candidate in reranked]

    def _retrieve_candidates(
        self,
        *,
        query_text: str,
        topn: int,
        initial_k: int,
    ) -> list[NormalizationCandidate]:
        if self.encoder is None or self.dictionary_embeddings is None:
            self.prepare()
        assert self.encoder is not None
        assert self.dictionary_embeddings is not None

        if not self.term_entries:
            return []

        initial_k = min(max(initial_k, topn), len(self.term_entries))
        query_rep = encode_texts(
            self.encoder,
            [query_text],
            batch_size=self.batch_size,
        )
        sims = cosine_similarity_matrix(query_rep, self.dictionary_embeddings)[0]
        top_indices = np.argsort(sims)[-initial_k:][::-1]

        best_by_concept: dict[str, NormalizationCandidate] = {}
        for idx in top_indices:
            entry = self.term_entries[int(idx)]
            score = float(sims[int(idx)])
            previous = best_by_concept.get(entry.identifier)
            if previous is not None and score <= previous.embedding_score:
                continue
            best_by_concept[entry.identifier] = NormalizationCandidate(
                identifier=entry.identifier,
                name=entry.raw_name,
                preferred_label=entry.preferred_label,
                embedding_score=score,
                final_score=score,
                is_preferred=entry.is_preferred,
                matched_alias=entry.raw_name,
            )

        candidates = sorted(
            best_by_concept.values(),
            key=lambda item: item.embedding_score,
            reverse=True,
        )
        return candidates[:topn]

    def _rerank_candidates(
        self,
        query_text: str,
        candidates: list[NormalizationCandidate],
    ) -> list[NormalizationCandidate]:
        query_norm = normalize_text(query_text)
        reranked: list[NormalizationCandidate] = []

        for candidate in candidates:
            meta = self.concept_metadata.get(candidate.identifier)
            names = set(meta.names) if meta else {candidate.name, candidate.preferred_label}
            preferred_label = meta.preferred_label if meta else candidate.preferred_label

            best_name_overlap = 0.0
            exact_synonym_match = 0.0
            best_parenthetical = 0.0
            best_seq = 0.0

            for name in names:
                if query_norm == normalize_text(name):
                    exact_synonym_match = 1.0
                best_name_overlap = max(best_name_overlap, token_jaccard(query_text, name))
                best_parenthetical = max(
                    best_parenthetical,
                    has_parenthetical_relation(query_text, name),
                )
                best_seq = max(best_seq, sequence_ratio(query_text, name))

            preferred_overlap = token_jaccard(query_text, preferred_label)
            final_score = (
                1.00 * candidate.embedding_score
                + 0.35 * exact_synonym_match
                + 0.20 * best_name_overlap
                + 0.15 * preferred_overlap
                + 0.10 * best_parenthetical
                + 0.05 * best_seq
                + (0.03 if candidate.is_preferred else 0.0)
            )

            candidate.final_score = float(final_score)
            candidate.exact_synonym_match = exact_synonym_match
            candidate.token_overlap = max(best_name_overlap, preferred_overlap)
            candidate.preferred_overlap = preferred_overlap
            candidate.parenthetical_match = best_parenthetical
            candidate.sequence_ratio = best_seq
            reranked.append(candidate)

        reranked.sort(
            key=lambda item: (item.final_score, item.embedding_score),
            reverse=True,
        )
        return reranked

    def _preferred_label_for(self, identifier: str, *, fallback: str = "") -> str:
        meta = self.concept_metadata.get(identifier)
        if meta is not None and meta.preferred_label:
            return meta.preferred_label
        return fallback


def _strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _children(element: ET.Element, tag: str) -> list[ET.Element]:
    return [child for child in list(element) if _strip_namespace(child.tag) == tag]


def _child(element: ET.Element, tag: str) -> Optional[ET.Element]:
    matches = _children(element, tag)
    return matches[0] if matches else None


def _child_text(element: ET.Element, tag: str, default: str = "") -> str:
    item = _child(element, tag)
    if item is None or item.text is None:
        return default
    return item.text


def _infons(element: ET.Element) -> dict[str, str]:
    output: dict[str, str] = {}
    for infon in _children(element, "infon"):
        key = infon.attrib.get("key")
        if key:
            output[key] = infon.text or ""
    return output


def _set_infon(element: ET.Element, key: str, value: Any) -> None:
    for infon in _children(element, "infon"):
        if infon.attrib.get("key") == key:
            infon.text = str(value)
            return
    infon = ET.Element("infon", {"key": key})
    infon.text = str(value)
    element.insert(0, infon)


def _document_id(document: ET.Element) -> str:
    return _child_text(document, "id", default="")


def _get_document_key(document: ET.Element, passage: ET.Element) -> str:
    passage_infons = _infons(passage)
    key = (
        passage_infons.get("article-id_pmid")
        or _document_id(document)
        or passage_infons.get("passage_id")
    )
    if not key:
        raise ValueError("Could not determine a stable document key for normalization")
    return str(key)


def _annotation_is_processable(annotation: ET.Element) -> bool:
    text = _child_text(annotation, "text", default="")
    if not text.strip():
        return False
    annotation_type = _infons(annotation).get("type", "")
    return annotation_type != "cell_vague"


def collect_mentions_from_bioc(
    input_xml: str | Path,
) -> tuple[set[tuple[str, str]], dict[str, str], int]:
    """
    Collect unique normalized mention texts and document text from BioC XML.
    """
    tree = ET.parse(input_xml)
    root = tree.getroot()

    mentions: set[tuple[str, str]] = set()
    document_text_by_key: dict[str, str] = {}
    annotation_count = 0

    for document in root.iter():
        if _strip_namespace(document.tag) != "document":
            continue
        document_key: Optional[str] = None
        passage_texts: list[str] = []
        for passage_index, passage in enumerate(_children(document, "passage")):
            passage_infons = _infons(passage)
            if str(passage_infons.get("annotatable", "true")).casefold() == "false":
                continue
            passage_text = _child_text(passage, "text", default="")
            if not passage_text:
                continue
            if document_key is None:
                document_key = _get_document_key(document, passage)
            passage_texts.append(passage_text)
            for annotation in _children(passage, "annotation"):
                if not _annotation_is_processable(annotation):
                    continue
                annotation_count += 1
                mention_text = _child_text(annotation, "text", default="")
                mentions.add((document_key, normalize_text(mention_text)))
        if document_key is not None:
            document_text_by_key[document_key] = "\n".join(passage_texts)

    return mentions, document_text_by_key, annotation_count


def _target_document_keys_for_ab3p(
    mentions: Iterable[tuple[str, str]],
    abbreviation_lookup: AbbreviationLookup,
) -> set[str]:
    if not abbreviation_lookup or not abbreviation_lookup.ambiguous_candidates:
        return set()
    ambiguous_keys = set(abbreviation_lookup.ambiguous_candidates.keys())
    target_document_keys: set[str] = set()
    for document_key, mention_text in mentions:
        mention_key = normalize_abbreviation_key(mention_text)
        if mention_key in ambiguous_keys or is_abbreviation_like(mention_text):
            target_document_keys.add(document_key)
    return target_document_keys


def _apply_results_to_tree(
    tree: ET.ElementTree,
    results: dict[tuple[Optional[str], str], NormalizationResult],
    *,
    model_label: str,
) -> tuple[int, int, int, Counter]:
    processed_annotations = 0
    matched_annotations = 0
    missing_annotations = 0
    stats: Counter = Counter()

    root = tree.getroot()
    for document in root.iter():
        if _strip_namespace(document.tag) != "document":
            continue
        for passage_index, passage in enumerate(_children(document, "passage")):
            passage_infons = _infons(passage)
            if str(passage_infons.get("annotatable", "true")).casefold() == "false":
                continue
            document_key = _get_document_key(document, passage)
            for annotation_index, annotation in enumerate(_children(passage, "annotation")):
                if not _annotation_is_processable(annotation):
                    continue
                processed_annotations += 1
                mention_text = _child_text(annotation, "text", default="")
                normalized_text = normalize_text(mention_text)
                result = results.get((document_key, normalized_text)) or results.get(
                    (None, normalized_text)
                )
                best = result.best if result else None
                if best is None:
                    missing_annotations += 1
                    stats["no_candidate"] += 1
                    continue

                matched_annotations += 1
                stats[best.source] += 1
                _set_infon(annotation, f"{model_label}_id_0", best.identifier)
                _set_infon(annotation, f"{model_label}_identifier_name_0", best.name)
                _set_infon(annotation, f"{model_label}_identifier_score_0", best.final_score)
                _set_infon(annotation, f"{model_label}_embedding_score_0", best.embedding_score)
                _set_infon(annotation, f"{model_label}_preferred_label_0", best.preferred_label)
                _set_infon(annotation, f"{model_label}_match_source", best.source)
                if best.abbreviation_method:
                    _set_infon(annotation, f"{model_label}_abbreviation_method", best.abbreviation_method)
                if best.expanded_long_form:
                    _set_infon(annotation, f"{model_label}_expanded_long_form", best.expanded_long_form)
                if best.ab3p_method:
                    _set_infon(annotation, f"{model_label}_ab3p_method", best.ab3p_method)
                if best.ab3p_matched_key:
                    _set_infon(annotation, f"{model_label}_ab3p_matched_key", best.ab3p_matched_key)
                if best.ab3p_match_score is not None:
                    _set_infon(annotation, f"{model_label}_ab3p_match_score", best.ab3p_match_score)

    return processed_annotations, matched_annotations, missing_annotations, stats


def normalize_mentions(
    mentions: Iterable[str | tuple[str, str]],
    *,
    ontology_path: str | Path,
    model_path: str | Path = DEFAULT_NEN_MODEL,
    abbreviations_path: str | Path | Iterable[str | Path] | None = None,
    disable_abbreviations: bool = False,
    document_abbreviation_lookup: Optional[dict[str, dict[str, str]]] = None,
    batch_size: int = 128,
    device: Optional[str] = None,
    trust_remote_code: bool = False,
    verbose: bool = True,
) -> dict[tuple[Optional[str], str], NormalizationResult]:
    """Normalize mention strings without reading/writing BioC XML."""
    linker = CellOntologyLinker.from_files(
        ontology_path=ontology_path,
        model_path=model_path,
        abbreviations_path=abbreviations_path,
        disable_abbreviations=disable_abbreviations,
        document_abbreviation_lookup=document_abbreviation_lookup,
        batch_size=batch_size,
        device=device,
        trust_remote_code=trust_remote_code,
        verbose=verbose,
    )
    return linker.link_mentions(mentions)


def normalize_bioc(
    input_xml: str | Path,
    output_xml: str | Path,
    *,
    cell_types: str | Path | None = None,
    ontology_path: str | Path | None = None,
    model_path: str | Path = DEFAULT_NEN_MODEL,
    abbreviations: str | Path | Iterable[str | Path] | None = None,
    abbreviations_path: str | Path | Iterable[str | Path] | None = None,
    disable_abbreviations: bool = False,
    abbr_verbose: bool = False,
    el_warmup_runs: int = 0,
    batch_size: int = 128,
    device: Optional[str] = None,
    trust_remote_code: bool = False,
    model_label: Optional[str] = None,
    write_runtime_summary: bool = True,
) -> Path:
    """
    Normalize cell-type annotations in a BioC XML file.

    Parameters use both ``cell_types`` and ``ontology_path`` for compatibility
    with the old code and the new package API.
    """
    input_xml = Path(input_xml)
    output_xml = Path(output_xml)
    if not input_xml.is_file():
        raise FileNotFoundError(f"Missing input BioC XML: {input_xml}")

    ontology = Path(ontology_path or cell_types or default_ontology_path())
    if not ontology.is_file():
        raise FileNotFoundError(f"Missing Cell Ontology JSONL: {ontology}")

    if abbreviations_path is not None:
        abbreviation_resource = abbreviations_path
    elif abbreviations is not None:
        abbreviation_resource = abbreviations
    else:
        default_abbr = default_abbreviations_path()
        abbreviation_resource = default_abbr if default_abbr.is_file() else None

    if disable_abbreviations:
        abbreviation_resource = None

    model_name_or_path, inferred_model_label = resolve_model_reference(model_path)
    model_label = _clean_model_label(model_label or inferred_model_label)

    output_xml.parent.mkdir(parents=True, exist_ok=True)

    if abbr_verbose:
        print(f"Collecting mentions from {input_xml}")
    mentions, document_text_by_key, annotation_count = collect_mentions_from_bioc(input_xml)

    abbreviation_lookup = AbbreviationLookup()
    if abbreviation_resource not in [None, "", "."]:
        abbreviation_lookup = load_abbreviation_identifier_lookup(
            abbreviation_resource,
            verbose=abbr_verbose,
        )

    target_document_keys = _target_document_keys_for_ab3p(mentions, abbreviation_lookup)
    document_abbreviation_lookup = build_document_abbreviation_lookup(
        document_text_by_key,
        target_document_keys,
        verbose=abbr_verbose,
    )

    linker = CellOntologyLinker.from_files(
        ontology_path=ontology,
        model_path=model_name_or_path,
        abbreviations_path=None,
        disable_abbreviations=True,
        document_abbreviation_lookup=document_abbreviation_lookup,
        batch_size=batch_size,
        device=device,
        trust_remote_code=trust_remote_code,
        verbose=abbr_verbose,
    )
    linker.abbreviation_lookup = abbreviation_lookup
    if abbreviation_lookup and abbreviation_lookup.all_keys:
        linker.abbreviation_embeddings = None
        linker.prepare()

    if el_warmup_runs < 0:
        raise ValueError("el_warmup_runs must be >= 0")

    mentions_list = sorted(mentions)
    for warmup_idx in range(el_warmup_runs):
        if abbr_verbose:
            print(f"EL warmup run {warmup_idx + 1}/{el_warmup_runs}")
        _ = linker.link_mentions(mentions_list)

    start_time = time.perf_counter()
    results = linker.link_mentions(mentions_list)
    elapsed = time.perf_counter() - start_time

    tree = ET.parse(input_xml)
    processed, matched, missing, stats = _apply_results_to_tree(
        tree,
        results,
        model_label=model_label,
    )
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)

    runtime = RuntimeSummary(
        elapsed_seconds=float(elapsed),
        total_unique_mentions=len(mentions_list),
        total_annotations=annotation_count,
        model_name=model_label,
        stats={str(key): int(value) for key, value in stats.items()},
    )

    if write_runtime_summary:
        runtime_path = output_xml.resolve().parent / EL_RUNTIME_SUMMARY_FILENAME
        current: dict[str, Any] = {}
        if runtime_path.is_file():
            try:
                current = json.loads(runtime_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                current = {}
        current[model_label] = runtime.to_dict()
        current[model_label]["processed_annotations"] = processed
        current[model_label]["matched_annotations"] = matched
        current[model_label]["missing_annotations"] = missing
        runtime_path.write_text(json.dumps(current, indent=2), encoding="utf-8")

    if abbr_verbose:
        print(
            "Processed {} annotations, matched {}, missing {}".format(
                processed, matched, missing
            )
        )
        print(f"Wrote normalized BioC XML to {output_xml}")

    return output_xml


def paths_to_filenames(
    input_paths: str | Path | Sequence[str | Path],
    output_paths: str | Path | Sequence[str | Path],
) -> tuple[list[Path], list[Path]]:
    """Expand file-or-directory input/output arguments."""
    if isinstance(input_paths, (str, Path)):
        inputs = [Path(input_paths)]
    else:
        inputs = [Path(item) for item in input_paths]

    if isinstance(output_paths, (str, Path)):
        outputs = [Path(output_paths)]
    else:
        outputs = [Path(item) for item in output_paths]

    if len(inputs) != len(outputs):
        raise ValueError("input_paths and output_paths must have the same length")

    new_inputs: list[Path] = []
    new_outputs: list[Path] = []
    for input_path, output_path in zip(inputs, outputs):
        if input_path.is_dir() and output_path.is_dir():
            for child in sorted(input_path.iterdir()):
                if child.is_file():
                    new_inputs.append(child)
                    new_outputs.append(output_path / child.name)
        elif input_path.is_file() and not output_path.is_dir():
            new_inputs.append(input_path)
            new_outputs.append(output_path)
        else:
            raise ValueError("both input and output path must be either directory or file")
    return new_inputs, new_outputs


def main(
    term_filename: str | Path,
    abbr_paths: str | Path | Iterable[str | Path] | None,
    input_paths: str | Path | Sequence[str | Path],
    output_paths: str | Path | Sequence[str | Path],
    model_names: dict[str, str | Path],
    *,
    abbr_verbose: bool = True,
    el_warmup_runs: int = 0,
    batch_size: int = 128,
    device: Optional[str] = None,
    trust_remote_code: bool = False,
) -> None:
    """
    Backward-compatible entry point similar to the old normalize.py ``main``.
    """
    input_files, output_files = paths_to_filenames(input_paths, output_paths)

    for input_file, output_file in zip(input_files, output_files):
        working_input = input_file
        for model_label, model_path in model_names.items():
            normalize_bioc(
                input_xml=working_input,
                output_xml=output_file,
                cell_types=term_filename,
                model_path=model_path,
                abbreviations=abbr_paths,
                disable_abbreviations=abbr_paths in [None, "", "."],
                abbr_verbose=abbr_verbose,
                el_warmup_runs=el_warmup_runs,
                batch_size=batch_size,
                device=device,
                trust_remote_code=trust_remote_code,
                model_label=model_label,
            )
            working_input = output_file


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize cell-type BioC annotations to Cell Ontology IDs.")
    parser.add_argument("input_xml", help="Input BioC XML file or directory.")
    parser.add_argument("output_xml", help="Output BioC XML file or directory.")
    parser.add_argument(
        "--cell-types",
        "--ontology-path",
        dest="cell_types",
        default=str(default_ontology_path()),
        help="Cell Ontology JSONL resource.",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_NEN_MODEL,
        help=f"Normalization model name or local path. Default: {DEFAULT_NEN_MODEL}",
    )
    parser.add_argument(
        "--abbreviations",
        default=None,
        help="Abbreviation TSV. Defaults to package resource when available.",
    )
    parser.add_argument(
        "--disable-abbreviations",
        action="store_true",
        help="Disable abbreviation dictionary and document-level abbreviation handling.",
    )
    parser.add_argument("--abbr-verbose", action="store_true", help="Print abbreviation details.")
    parser.add_argument("--el-warmup-runs", type=int, default=0, help="Warmup runs before timing.")
    parser.add_argument("--batch-size", type=int, default=128, help="Encoder batch size.")
    parser.add_argument("--device", default=None, help="Device, for example 'cuda' or 'cpu'.")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args(argv)


def cli_main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    model_name_or_path, model_label = resolve_model_reference(args.model_path)
    main(
        term_filename=args.cell_types,
        abbr_paths=None if args.disable_abbreviations else args.abbreviations,
        input_paths=args.input_xml,
        output_paths=args.output_xml,
        model_names={model_label: model_name_or_path},
        abbr_verbose=args.abbr_verbose,
        el_warmup_runs=args.el_warmup_runs,
        batch_size=args.batch_size,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
