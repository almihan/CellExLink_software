"""Quick start for CellExLink BioC XML input/output.

This example demonstrates the BioC XML workflow:

1. recognize_bioc(): NER only
2. normalize_bioc(): NEN only on existing/gold mentions
3. extract_bioc(): end-to-end NER + NEN
"""

from __future__ import annotations

from pathlib import Path

from cellexlink import CellExLinkPipeline


def main() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    pipe = CellExLinkPipeline.from_pretrained(
        ner_model="models/CellExLink-bioformer16L",
        nen_model="models/CellExLink-Sapbert",
        ontology_path="src/cellexlink/resources/cell_ontology_v2025-12-17.jsonl",
        abbreviations_path="src/cellexlink/resources/abbreviations.tsv",
    )

    pipe.recognize_bioc(
        input_xml="examples/sample_input.xml",
        output_xml=output_dir / "sample.ner.xml",
    )

    pipe.normalize_bioc(
        input_xml="examples/sample_gold_spans.xml",
        output_xml=output_dir / "sample.normalized.xml",
    )

    pipe.extract_bioc(
        input_xml="examples/sample_input.xml",
        output_xml=output_dir / "sample.end_to_end.xml",
        ner_output_xml=output_dir / "sample.end_to_end.ner.xml",
    )

    print(f"Wrote outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()