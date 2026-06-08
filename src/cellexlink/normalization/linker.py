"""Cell Ontology linker for CellExLink.

This file is the reproducibility-first normalizer.  It keeps the NEN behavior
of the original CellExLink code while presenting a small reusable API for the
SoftwareX package.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Optional
from xml.etree import ElementTree as ET

import numpy as np

from .abbreviations import (
    AbbreviationLookup,
    abbreviation_sequence_ratio,
    abbreviation_threshold_for,
    abbreviation_variant_keys,
    ab3p_fuzzy_threshold_for,
    build_document_abbreviation_lookup,
    default_abbreviations_path,
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
from .stemmer import plural_normalize_text

DEFAULT_NEN_MODEL = "almire/CellExLink-Sapbert"
DEFAULT_TOPN = 1
AMBIGUOUS_TOPN = 5


@dataclass(slots=True)
class EncoderHandle:
    kind: str
    model: Any
    tokenizer: Any = None
    device: str | None = None


@dataclass(slots=True, frozen=True)
class MentionRecord:
    mention_text: str
    document_key: str | None = None


@dataclass(slots=True)
class NormalizationCandidate:
    identifier: str
    name: str
    preferred_label: str
    embedding_score: float
    final_score: float
    source: str = "model_normal"
    is_preferred: bool = False
    matched_alias: str | None = None
    exact_synonym_match: float = 0.0
    token_overlap: float = 0.0
    preferred_overlap: float = 0.0
    parenthetical_match: float = 0.0
    sequence_ratio: float = 0.0
    abbreviation_method: str | None = None
    expanded_long_form: str | None = None
    ab3p_method: str | None = None
    ab3p_matched_key: str | None = None
    ab3p_match_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "name": self.name,
            "preferred_label": self.preferred_label,
            "embedding_score": self.embedding_score,
            "final_score": self.final_score,
            "source": self.source,
            "is_preferred": self.is_preferred,
            "matched_alias": self.matched_alias,
            "exact_synonym_match": self.exact_synonym_match,
            "token_overlap": self.token_overlap,
            "preferred_overlap": self.preferred_overlap,
            "parenthetical_match": self.parenthetical_match,
            "sequence_ratio": self.sequence_ratio,
            "abbreviation_method": self.abbreviation_method,
            "expanded_long_form": self.expanded_long_form,
            "ab3p_method": self.ab3p_method,
            "ab3p_matched_key": self.ab3p_matched_key,
            "ab3p_match_score": self.ab3p_match_score,
        }


@dataclass(slots=True)
class NormalizationResult:
    mention_text: str
    normalized_text: str
    document_key: str | None
    candidates: list[NormalizationCandidate] = field(default_factory=list)

    @property
    def best(self) -> NormalizationCandidate | None:
        return self.candidates[0] if self.candidates else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mention_text": self.mention_text,
            "normalized_text": self.normalized_text,
            "document_key": self.document_key,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(slots=True)
class NormalizationRuntimeSummary:
    elapsed_seconds: float
    total_unique_mentions: int

    @property
    def mentions_per_second(self) -> float | None:
        if self.elapsed_seconds <= 0:
            return None
        return self.total_unique_mentions / self.elapsed_seconds

    @property
    def ms_per_mention(self) -> float | None:
        if self.total_unique_mentions <= 0:
            return None
        return (self.elapsed_seconds * 1000.0) / self.total_unique_mentions

    def to_dict(self) -> dict[str, Any]:
        return {
            "elapsed_seconds": self.elapsed_seconds,
            "total_unique_mentions": self.total_unique_mentions,
            "mentions_per_second": self.mentions_per_second,
            "ms_per_mention": self.ms_per_mention,
        }


def _casefold_normalize_text(text: object) -> str:
    """Small lexical normalizer used only for reranking features."""

    return " ".join(str(text).casefold().split())


def token_jaccard(left: object, right: object) -> float:
    left_tokens = set(_casefold_normalize_text(left).split())
    right_tokens = set(_casefold_normalize_text(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def sequence_ratio(left: object, right: object) -> float:
    return SequenceMatcher(
        None, _casefold_normalize_text(left), _casefold_normalize_text(right)
    ).ratio()


def has_parenthetical_relation(left: object, right: object) -> float:
    left_text = str(left)
    right_text = str(right)

    if "(" in left_text and ")" in left_text:
        inside = re.findall(r"\(([^()]*)\)", left_text)
        if any(_casefold_normalize_text(item) == _casefold_normalize_text(right_text) for item in inside):
            return 1.0

    if "(" in right_text and ")" in right_text:
        inside = re.findall(r"\(([^()]*)\)", right_text)
        if any(_casefold_normalize_text(item) == _casefold_normalize_text(left_text) for item in inside):
            return 1.0

    return 0.0


def is_sentence_transformers_model(model_name_or_path: str | Path) -> bool:
    """Return True only for local SentenceTransformer directories.

    This intentionally follows the original CellExLink normalizer: a model is
    treated as SentenceTransformer only when a local ``modules.json`` exists.
    Plain Hugging Face checkpoints are loaded with AutoTokenizer/AutoModel.
    """

    model_path = Path(model_name_or_path)
    return model_path.is_dir() and (model_path / "modules.json").is_file()


def load_encoder(
    model_name_or_path: str | Path,
    *,
    device: str | None = None,
    trust_remote_code: bool = False,
) -> EncoderHandle:
    """Load the SapBERT encoder using the original CellExLink decision rule."""

    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    if is_sentence_transformers_model(model_name_or_path):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(str(model_name_or_path))
        model = model.to(device)
        return EncoderHandle(kind="sentence_transformers", model=model, device=device)

    from transformers import AutoModel, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_name_or_path), trust_remote_code=trust_remote_code
    )
    model = AutoModel.from_pretrained(
        str(model_name_or_path), trust_remote_code=trust_remote_code
    )
    model = model.to(torch.device(device))
    model.eval()
    return EncoderHandle(
        kind="transformers", model=model, tokenizer=tokenizer, device=device
    )


def encode_texts(
    encoder: EncoderHandle,
    texts: list[str],
    *,
    batch_size: int = 128,
    max_length: int = 32,
) -> np.ndarray:
    """Encode names/queries with original-compatible settings.

    For plain transformers checkpoints, this uses fixed ``padding='max_length'``
    and the CLS vector ``output[0][:, 0, :]`` as in the original NEN code.
    """

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
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            toks = {key: value.to(device) for key, value in toks.items()}
            output = model(**toks)
            cls_rep = output[0][:, 0, :]
            all_reps.append(cls_rep.cpu().detach().numpy().astype(np.float32))

    return np.concatenate(all_reps, axis=0)


def cosine_similarity_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.size == 0 or right.size == 0:
        return np.zeros((left.shape[0], right.shape[0]), dtype=np.float32)

    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True)
    left_norm[left_norm == 0.0] = 1.0
    right_norm[right_norm == 0.0] = 1.0
    return (left / left_norm) @ (right / right_norm).T


class CellOntologyLinker:
    """Reusable Cell Ontology linker with original-compatible NEN behavior."""

    def __init__(
        self,
        *,
        model_path: str | Path = DEFAULT_NEN_MODEL,
        term_entries: list[TermEntry],
        concept_metadata: dict[str, ConceptMetadata],
        abbreviation_lookup: AbbreviationLookup | None = None,
        document_abbreviation_lookup: dict[str, dict[str, str]] | None = None,
        batch_size: int = 128,
        topn: int = DEFAULT_TOPN,
        ambiguous_topn: int = AMBIGUOUS_TOPN,
        device: str | None = None,
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

        self.encoder: EncoderHandle | None = None
        self.dictionary_embeddings: np.ndarray | None = None
        self.abbreviation_embeddings: np.ndarray | None = None
        self._query_cache: dict[tuple[str, str], list[NormalizationCandidate]] = {}
        self._abbreviation_result_cache: dict[str, dict[str, Any] | None] = {}
        self._mention_embedding_cache: dict[str, np.ndarray] = {}

    @classmethod
    def from_files(
        cls,
        *,
        ontology_path: str | Path,
        model_path: str | Path = DEFAULT_NEN_MODEL,
        abbreviations_path: str | Path | Iterable[str | Path] | None = None,
        disable_abbreviations: bool = False,
        document_text_by_key: dict[str, str] | None = None,
        target_document_keys: Iterable[str] | None = None,
        batch_size: int = 128,
        topn: int = DEFAULT_TOPN,
        ambiguous_topn: int = AMBIGUOUS_TOPN,
        device: str | None = None,
        trust_remote_code: bool = False,
        verbose: bool = True,
    ) -> "CellOntologyLinker":
        term_entries, concept_metadata = load_cell_ontology_terms(ontology_path)

        abbreviation_lookup = AbbreviationLookup()
        document_abbreviation_lookup: dict[str, dict[str, str]] = {}
        if not disable_abbreviations:
            if abbreviations_path is None:
                default_path = default_abbreviations_path()
                abbreviations_path = default_path if default_path.is_file() else None
            abbreviation_lookup = load_abbreviation_identifier_lookup(
                abbreviations_path, verbose=verbose
            )
            if document_text_by_key and target_document_keys:
                document_abbreviation_lookup = build_document_abbreviation_lookup(
                    document_text_by_key,
                    target_document_keys,
                    verbose=verbose,
                )

        return cls(
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

    def prepare(self) -> None:
        if self.encoder is None:
            if self.verbose:
                print(f"Loading NEN encoder: {self.model_path}")
            self.encoder = load_encoder(
                self.model_path,
                device=self.device,
                trust_remote_code=self.trust_remote_code,
            )

        if self.dictionary_embeddings is None:
            if self.verbose:
                print("Encoding Cell Ontology aliases")
            dictionary_names = [entry.name for entry in self.term_entries]
            self.dictionary_embeddings = encode_texts(
                self.encoder,
                dictionary_names,
                batch_size=self.batch_size,
            )

        if (
            self.abbreviation_embeddings is None
            and self.abbreviation_lookup
            and self.abbreviation_lookup.all_keys
        ):
            if self.verbose:
                print("Encoding abbreviation short forms")
            self.abbreviation_embeddings = encode_texts(
                self.encoder,
                self.abbreviation_lookup.all_keys,
                batch_size=self.batch_size,
            )

    def link_mention(
        self,
        mention_text: str,
        *,
        document_key: str | None = None,
        topn: int | None = None,
    ) -> NormalizationResult:
        if self.encoder is None or self.dictionary_embeddings is None:
            self.prepare()

        assert self.encoder is not None
        assert self.dictionary_embeddings is not None

        normalized_text = plural_normalize_text(mention_text)
        requested_topn = topn or self.topn

        if is_abbreviation_like(mention_text) and self.abbreviation_lookup:
            abbreviation_result = self._resolve_abbreviation(
                mention_text=mention_text,
                document_key=document_key,
            )
            if abbreviation_result is not None:
                if abbreviation_result["type"] == "direct":
                    identifier = abbreviation_result["identifier"]
                    candidate = NormalizationCandidate(
                        identifier=identifier,
                        name=abbreviation_result["short_form"],
                        preferred_label=self._preferred_label_for(
                            identifier, fallback=abbreviation_result["short_form"]
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
                    query_text = plural_normalize_text(abbreviation_result["expanded_long_form"])
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
                            candidates=candidates[:requested_topn],
                        )
                # For ambiguous_unresolved, fall through to normal model linking.

        candidates = self._retrieve_and_rerank(
            query_text=normalized_text,
            topn=requested_topn,
            initial_k=max(requested_topn * 10, 20),
            cache_prefix="normal",
        )
        for candidate in candidates:
            candidate.source = "model_normal"

        return NormalizationResult(
            mention_text=mention_text,
            normalized_text=normalized_text,
            document_key=document_key,
            candidates=candidates,
        )

    def link_many(self, mentions: Iterable[str | MentionRecord]) -> list[NormalizationResult]:
        results: list[NormalizationResult] = []
        for item in mentions:
            if isinstance(item, MentionRecord):
                results.append(
                    self.link_mention(item.mention_text, document_key=item.document_key)
                )
            else:
                results.append(self.link_mention(str(item)))
        return results

    def _find_best_abbreviation_key(self, mention_text: str) -> dict[str, Any] | None:
        if not self.abbreviation_lookup or self.abbreviation_embeddings is None:
            return None

        cache_key = mention_text
        if cache_key in self._abbreviation_result_cache:
            return self._abbreviation_result_cache[cache_key]

        mention_key = normalize_abbreviation_key(mention_text)
        if not mention_key:
            self._abbreviation_result_cache[cache_key] = None
            return None

        threshold = abbreviation_threshold_for(mention_text)

        if threshold == 1.0:
            if mention_key in self.abbreviation_lookup.key_to_candidates:
                result = {
                    "matched_key": mention_key,
                    "score": 1.0,
                    "method": "exact_short_abbreviation",
                }
                self._abbreviation_result_cache[cache_key] = result
                return result
            self._abbreviation_result_cache[cache_key] = None
            return None

        if mention_key in self._mention_embedding_cache:
            query_rep = self._mention_embedding_cache[mention_key]
        else:
            assert self.encoder is not None
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
            self._abbreviation_result_cache[cache_key] = result
            return result

        self._abbreviation_result_cache[cache_key] = None
        return None

    def _find_ab3p_long_form_with_fallback(
        self, doc_lookup: dict[str, str], matched_key: str
    ) -> tuple[str | None, str | None, str | None, float | None]:
        if not doc_lookup:
            return None, None, None, None

        exact_hit = doc_lookup.get(matched_key)
        if exact_hit:
            return exact_hit, "ab3p_exact_key", matched_key, 1.0

        for variant_key in abbreviation_variant_keys(matched_key):
            if variant_key == matched_key:
                continue
            variant_hit = doc_lookup.get(variant_key)
            if variant_hit:
                return variant_hit, "ab3p_exact_variant", variant_key, 1.0

        threshold = ab3p_fuzzy_threshold_for(matched_key)
        best_key: str | None = None
        best_score = -1.0
        for doc_key in doc_lookup.keys():
            score = abbreviation_sequence_ratio(matched_key, doc_key)
            if score > best_score:
                best_score = score
                best_key = doc_key

        if best_key is not None and best_score >= threshold:
            return doc_lookup[best_key], "ab3p_fuzzy_shortform", best_key, float(best_score)

        return None, None, None, None

    def _resolve_abbreviation(
        self,
        *,
        mention_text: str,
        document_key: str | None,
    ) -> dict[str, Any] | None:
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

        doc_lookup = self.document_abbreviation_lookup.get(document_key or "", {})
        expanded_long_form, ab3p_method, ab3p_matched_key, ab3p_score = (
            self._find_ab3p_long_form_with_fallback(doc_lookup, matched_key)
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

        # Original code used scipy cosine distance and stored -distance as the
        # embedding score.  Since cosine_distance = 1 - cosine_similarity, the
        # compatible score is cosine_similarity - 1.
        original_embedding_scores = sims - 1.0
        top_indices = np.argsort(original_embedding_scores)[-initial_k:][::-1]

        best_by_concept: dict[str, NormalizationCandidate] = {}
        for idx in top_indices:
            entry = self.term_entries[int(idx)]
            score = float(original_embedding_scores[int(idx)])
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
        query_norm = _casefold_normalize_text(query_text)
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
                if query_norm == _casefold_normalize_text(name):
                    exact_synonym_match = 1.0
                best_name_overlap = max(best_name_overlap, token_jaccard(query_text, name))
                best_parenthetical = max(
                    best_parenthetical, has_parenthetical_relation(query_text, name)
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


def normalize_mentions(
    mentions: Iterable[str | MentionRecord],
    *,
    ontology_path: str | Path | None = None,
    model_path: str | Path = DEFAULT_NEN_MODEL,
    abbreviations_path: str | Path | Iterable[str | Path] | None = None,
    disable_abbreviations: bool = False,
    document_text_by_key: dict[str, str] | None = None,
    batch_size: int = 128,
    topn: int = DEFAULT_TOPN,
    ambiguous_topn: int = AMBIGUOUS_TOPN,
    device: str | None = None,
    trust_remote_code: bool = False,
    verbose: bool = True,
) -> list[NormalizationResult]:
    mention_list = list(mentions)
    target_document_keys = [
        item.document_key
        for item in mention_list
        if isinstance(item, MentionRecord) and item.document_key is not None
    ]
    linker = CellOntologyLinker.from_files(
        ontology_path=ontology_path or default_ontology_path(),
        model_path=model_path,
        abbreviations_path=abbreviations_path,
        disable_abbreviations=disable_abbreviations,
        document_text_by_key=document_text_by_key,
        target_document_keys=target_document_keys,
        batch_size=batch_size,
        topn=topn,
        ambiguous_topn=ambiguous_topn,
        device=device,
        trust_remote_code=trust_remote_code,
        verbose=verbose,
    )
    return linker.link_many(mention_list)


def resolve_model_label(model_reference: str | Path) -> str:
    """Return the BioC infon prefix expected by the original evaluator."""

    text = str(model_reference)
    if "sapbert" in text.casefold():
        return "CellExLink-Sapbert"
    name = Path(text).name or text
    return name.replace("_", "-")


def normalize_bioc(
    input_xml: str | Path,
    output_xml: str | Path,
    *,
    model_path: str | Path = DEFAULT_NEN_MODEL,
    cell_types: str | Path | None = None,
    ontology_path: str | Path | None = None,
    abbreviations: str | Path | Iterable[str | Path] | None = None,
    abbreviations_path: str | Path | Iterable[str | Path] | None = None,
    disable_abbreviations: bool = False,
    batch_size: int = 128,
    topn: int = DEFAULT_TOPN,
    ambiguous_topn: int = AMBIGUOUS_TOPN,
    device: str | None = None,
    trust_remote_code: bool = False,
    model_label: str | None = None,
    verbose: bool = True,
) -> Path:
    """Normalize cell-type annotations in a BioC XML file.

    Parameters ``cell_types`` and ``abbreviations`` are kept for compatibility
    with the old scripts and the current SoftwareX pipeline wrapper.
    """

    input_xml = Path(input_xml)
    output_xml = Path(output_xml)
    if not input_xml.is_file():
        raise FileNotFoundError(f"Input BioC XML file does not exist: {input_xml}")

    ontology_resource = ontology_path or cell_types or default_ontology_path()
    abbreviation_resource = abbreviations_path if abbreviations_path is not None else abbreviations

    tree = ET.parse(input_xml)
    root = tree.getroot()

    annotation_refs, unique_mentions, document_text_by_key = _collect_bioc_mentions(root)

    target_document_keys = [record.document_key for record in unique_mentions if record.document_key]
    linker = CellOntologyLinker.from_files(
        ontology_path=ontology_resource,
        model_path=model_path,
        abbreviations_path=abbreviation_resource,
        disable_abbreviations=disable_abbreviations,
        document_text_by_key=document_text_by_key,
        target_document_keys=target_document_keys,
        batch_size=batch_size,
        topn=topn,
        ambiguous_topn=ambiguous_topn,
        device=device,
        trust_remote_code=trust_remote_code,
        verbose=verbose,
    )

    started = time.perf_counter()
    results = linker.link_many(unique_mentions)
    elapsed = time.perf_counter() - started

    result_by_key = {
        (result.document_key, result.normalized_text): result for result in results
    }
    label = model_label or resolve_model_label(model_path)

    for annotation, raw_text, document_key in annotation_refs:
        normalized_text = plural_normalize_text(raw_text)
        result = result_by_key.get((document_key, normalized_text))
        if result is None:
            continue
        _write_result_to_annotation(annotation, result, label=label, topn=topn)

    _set_collection_infon(root, "CellExLink_normalization_model", label)
    _set_collection_infon(
        root,
        "CellExLink_normalization_unique_mentions",
        str(len(unique_mentions)),
    )
    _set_collection_infon(
        root,
        "CellExLink_normalization_elapsed_seconds",
        f"{elapsed:.6f}",
    )

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    return output_xml


def _strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _children(element: ET.Element, tag: str) -> list[ET.Element]:
    return [child for child in list(element) if _strip_namespace(child.tag) == tag]


def _child(element: ET.Element, tag: str) -> ET.Element | None:
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


def _set_collection_infon(root: ET.Element, key: str, value: Any) -> None:
    if _strip_namespace(root.tag) == "collection":
        _set_infon(root, key, value)


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


def _is_annotatable(passage: ET.Element) -> bool:
    infons = _infons(passage)
    value = infons.get("annotatable", "true")
    return str(value).casefold() not in {"false", "0", "no"}


def _annotation_is_processable(annotation: ET.Element) -> bool:
    infons = _infons(annotation)
    if infons.get("type") == "cell_vague":
        return False
    return bool(_child_text(annotation, "text", default="").strip())


def _iter_documents(root: ET.Element) -> Iterable[ET.Element]:
    for element in root.iter():
        if _strip_namespace(element.tag) == "document":
            yield element


def _collect_bioc_mentions(
    root: ET.Element,
) -> tuple[list[tuple[ET.Element, str, str]], list[MentionRecord], dict[str, str]]:
    annotation_refs: list[tuple[ET.Element, str, str]] = []
    document_text_parts: dict[str, list[str]] = {}
    unique_seen: set[tuple[str, str]] = set()
    unique_mentions: list[MentionRecord] = []

    for document in _iter_documents(root):
        for passage in _children(document, "passage"):
            if not _is_annotatable(passage):
                continue
            document_key = _get_document_key(document, passage)
            passage_text = _child_text(passage, "text", default="")
            if passage_text:
                document_text_parts.setdefault(document_key, []).append(passage_text)

            for annotation in _children(passage, "annotation"):
                if not _annotation_is_processable(annotation):
                    continue
                raw_text = _child_text(annotation, "text", default="")
                normalized_text = plural_normalize_text(raw_text)
                key = (document_key, normalized_text)
                annotation_refs.append((annotation, raw_text, document_key))
                if key not in unique_seen:
                    unique_seen.add(key)
                    unique_mentions.append(
                        MentionRecord(
                            mention_text=normalized_text,
                            document_key=document_key,
                        )
                    )

    document_text_by_key = {
        key: "\n".join(parts) for key, parts in document_text_parts.items()
    }
    return annotation_refs, unique_mentions, document_text_by_key


def _write_result_to_annotation(
    annotation: ET.Element,
    result: NormalizationResult,
    *,
    label: str,
    topn: int,
) -> None:
    _set_infon(annotation, f"{label}_normalized_text", result.normalized_text)

    for rank, candidate in enumerate(result.candidates[:topn]):
        _set_infon(annotation, f"{label}_id_{rank}", candidate.identifier)
        _set_infon(annotation, f"{label}_identifier_name_{rank}", candidate.preferred_label)
        _set_infon(annotation, f"{label}_matched_alias_{rank}", candidate.matched_alias or candidate.name)
        _set_infon(annotation, f"{label}_identifier_score_{rank}", f"{candidate.final_score:.8f}")
        _set_infon(annotation, f"{label}_embedding_score_{rank}", f"{candidate.embedding_score:.8f}")

    best = result.best
    if best is not None:
        _set_infon(annotation, f"{label}_match_source", best.source)
        if best.abbreviation_method:
            _set_infon(annotation, f"{label}_abbreviation_method", best.abbreviation_method)
        if best.expanded_long_form:
            _set_infon(annotation, f"{label}_expanded_long_form", best.expanded_long_form)
        if best.ab3p_method:
            _set_infon(annotation, f"{label}_ab3p_method", best.ab3p_method)
        if best.ab3p_matched_key:
            _set_infon(annotation, f"{label}_ab3p_matched_key", best.ab3p_matched_key)
        if best.ab3p_match_score is not None:
            _set_infon(annotation, f"{label}_ab3p_match_score", f"{best.ab3p_match_score:.8f}")


__all__ = [
    "DEFAULT_NEN_MODEL",
    "DEFAULT_TOPN",
    "AMBIGUOUS_TOPN",
    "EncoderHandle",
    "MentionRecord",
    "NormalizationCandidate",
    "NormalizationResult",
    "NormalizationRuntimeSummary",
    "CellOntologyLinker",
    "is_sentence_transformers_model",
    "load_encoder",
    "encode_texts",
    "cosine_similarity_matrix",
    "token_jaccard",
    "sequence_ratio",
    "has_parenthetical_relation",
    "normalize_mentions",
    "normalize_bioc",
    "resolve_model_label",
]
