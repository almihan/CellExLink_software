"""Basic import tests for the CellExLink package."""

from __future__ import annotations


def test_top_level_import_exposes_public_api() -> None:
    import cellexlink

    assert hasattr(cellexlink, "CellExLinkPipeline")
    assert hasattr(cellexlink, "ExtractionResult")
    assert isinstance(cellexlink.__version__, str)
    assert cellexlink.__version__


def test_pipeline_can_be_constructed_without_loading_models() -> None:
    from cellexlink import CellExLinkPipeline

    pipeline = CellExLinkPipeline.from_pretrained(
        ner_model="dummy-ner-model",
        nen_model="dummy-nen-model",
    )

    assert pipeline.ner_model == "dummy-ner-model"
    assert pipeline.nen_model == "dummy-nen-model"


def test_lightweight_io_imports() -> None:
    from cellexlink.io import EntitySpan, PassageRecord, PredictedEntity

    span = EntitySpan(start=0, end=3, label="cell_type", text="SMC")
    record = PassageRecord(
        record_id=0,
        document_id="doc0",
        passage_id=0,
        passage_offset=0,
        text="SMC clusters were observed.",
        entities=[span],
    )
    predicted = span.to_absolute(record.passage_offset, document_id=record.document_id)

    assert isinstance(predicted, PredictedEntity)
    assert predicted.start == 0
    assert predicted.end == 3
    assert predicted.text == "SMC"
