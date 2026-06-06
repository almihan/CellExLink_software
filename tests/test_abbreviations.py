"""Tests for CellExLink abbreviation handling."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_normalize_abbreviation_key_removes_spaces_and_dashes() -> None:
    from cellexlink.normalization.abbreviations import normalize_abbreviation_key

    assert normalize_abbreviation_key(" S-MCs ") == "SMCs"
    assert normalize_abbreviation_key("CD8+ T") == "CD8+T"
    assert normalize_abbreviation_key("ECs") == "ECs"


def test_abbreviation_variants_include_singular_and_plural() -> None:
    from cellexlink.normalization.abbreviations import abbreviation_variant_keys

    assert abbreviation_variant_keys("SMC") == ["SMC", "SMCs"]
    assert abbreviation_variant_keys("SMCs") == ["SMCs", "SMC"]


def test_abbreviation_like_heuristic() -> None:
    from cellexlink.normalization.abbreviations import is_abbreviation_like

    assert is_abbreviation_like("SMC")
    assert is_abbreviation_like("CD8+ T cells")
    assert not is_abbreviation_like("mesothelial cell")


def test_load_abbreviation_identifier_lookup_direct_and_ambiguous(tmp_path: Path) -> None:
    from cellexlink.normalization.abbreviations import (
        classify_abbreviation_path,
        load_abbreviation_identifier_lookup,
    )

    abbr_tsv = tmp_path / "abbreviations.tsv"
    abbr_tsv.write_text(
        "short_form\tmatched_cl_id\n"
        "SMC\tCL:0000192\n"
        "SMCs\tCL:0000192\n"
        "EC\tCL:0000115\n"
        "EC\tCL:0000001\n",
        encoding="utf-8",
    )

    assert classify_abbreviation_path(abbr_tsv) == "short_form_identifier_tsv"

    lookup = load_abbreviation_identifier_lookup(abbr_tsv, verbose=False)

    assert lookup
    assert lookup.direct_lookup["SMC"][1] == "CL:0000192"
    assert lookup.direct_lookup["SMCs"][1] == "CL:0000192"
    assert "EC" in lookup.ambiguous_candidates
    assert {candidate.identifier for candidate in lookup.ambiguous_candidates["EC"]} == {
        "CL:0000115",
        "CL:0000001",
    }


def test_abbreviation_sequence_ratio_normalizes_before_comparing() -> None:
    from cellexlink.normalization.abbreviations import abbreviation_sequence_ratio

    assert abbreviation_sequence_ratio("S-MC", "SMC") == pytest.approx(1.0)
    assert abbreviation_sequence_ratio("CD8 T", "CD8T") == pytest.approx(1.0)
