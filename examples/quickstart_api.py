"""Minimal CellExLink Python API example.

Run after installing CellExLink and downloading model checkpoints.
"""

from cellexlink import CellExLinkPipeline

TEXT = "The mesothelial cell and SMC clusters formed the third population."

pipe = CellExLinkPipeline.from_pretrained(
    ner_model="models/CellExLink-bioformer16L",
    nen_model="models/CellExLink-Sapbert",
)

# 1. NER only
ner_results = pipe.recognize_text(TEXT)
print("NER")
for result in ner_results:
    print(result.to_dict())

# 2. NEN only: normalize known mentions without running NER
nen_results = pipe.normalize_mentions(
    ["mesothelial cell", "SMC"],
    document_text=TEXT,
)
print("NEN")
for result in nen_results:
    print(result.to_dict())

# 3. End-to-end extraction: NER followed by NEN
e2e_results = pipe.extract_text(TEXT)
print("END TO END")
for result in e2e_results:
    print(result.to_dict())
