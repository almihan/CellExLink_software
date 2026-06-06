# Input and output formats

CellExLink supports plain text, BioC XML, and JSONL formats. BioC XML is the recommended format for full document-level biomedical literature mining because it preserves document IDs, passage offsets, and annotation locations.

---

## 1. Plain text input

A plain text input file may contain one or more sentences or paragraphs:

```text
The mesothelial cell and SMC clusters formed the third population of the SS and SC.
CD8+ T cells were enriched in the tumor microenvironment after therapy.
```

Run:

```bash
cellexlink predict-text \
  --input examples/sample_input.txt \
  --output outputs/text_predictions.jsonl
```

Internally, CellExLink converts the text to a minimal BioC XML document and runs the same end-to-end pipeline used for BioC input.

---

## 2. Minimal BioC XML input

A minimal BioC XML file should look like this:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<collection>
  <source>CellExLink examples</source>
  <date></date>
  <key>cell_type_extraction_example</key>
  <document>
    <id>sample-doc-001</id>
    <passage>
      <infon key="type">abstract</infon>
      <offset>0</offset>
      <text>The mesothelial cell and SMC clusters formed the third population.</text>
    </passage>
  </document>
</collection>
```

Required fields:

```text
collection/document/id
collection/document/passage/offset
collection/document/passage/text
```

Recommended optional fields:

```text
collection/source
collection/date
collection/key
passage/infon[@key="type"]
```

---

## 3. BioC passage offsets

BioC offsets should be absolute document offsets.

Example:

```xml
<passage>
  <offset>100</offset>
  <text>CD8+ T cells were detected.</text>
</passage>
```

If CellExLink detects `CD8+ T cells` at character positions `0..12` inside the passage text, the BioC annotation location should be:

```xml
<location offset="100" length="12" />
```

because `100 + 0 = 100`.

This distinction matters when downstream tools use offsets to map predictions back to the original document.

---

## 4. BioC XML with gold annotations

For training or normalization-only evaluation, the input BioC XML may already contain annotations:

```xml
<annotation id="0">
  <infon key="type">cell_type</infon>
  <location offset="4" length="16" />
  <text>mesothelial cell</text>
</annotation>
```

CellExLink expects the annotation text and location to agree with the passage text whenever possible.

---

## 5. NER BioC output

The recognition step writes BioC annotations for detected cell-type spans.

Example:

```xml
<annotation id="CellExLink_NER_0">
  <infon key="type">cell_type</infon>
  <infon key="source">CellExLink</infon>
  <location offset="4" length="16" />
  <text>mesothelial cell</text>
</annotation>
```

The exact annotation ID prefix may vary, but each annotation should contain:

```text
infon key="type"
location offset
location length
text
```

---

## 6. Normalized BioC output

The normalization step adds Cell Ontology linking information to each recognized annotation.

Example:

```xml
<annotation id="CellExLink_NER_0">
  <infon key="type">cell_type</infon>
  <infon key="source">CellExLink</infon>
  <infon key="CellExLink-Sapbert_id_0">CL:0000077</infon>
  <infon key="CellExLink-Sapbert_identifier_name_0">mesothelial cell</infon>
  <infon key="CellExLink-Sapbert_identifier_score_0">0.991</infon>
  <infon key="CellExLink-Sapbert_preferred_label_0">mesothelial cell</infon>
  <infon key="CellExLink-Sapbert_match_source">dense_retrieval</infon>
  <location offset="4" length="16" />
  <text>mesothelial cell</text>
</annotation>
```

Common normalization infons:

| Infon suffix | Meaning |
|---|---|
| `_id_0` | Top predicted Cell Ontology identifier. |
| `_identifier_name_0` | Matched ontology alias or concept name. |
| `_identifier_score_0` | Score assigned by the linker. |
| `_preferred_label_0` | Preferred Cell Ontology label when available. |
| `_match_source` | Matching route, such as abbreviation, long-form recovery, or dense retrieval. |

The model-name prefix may change if you use a different normalization model. For example:

```text
CellExLink-Sapbert_id_0
custom-linker_id_0
```

Downstream code should search by suffix when possible.

---

## 7. JSONL output from plain text

`predict-text` writes one JSON object per predicted mention:

```json
{"document_id":"doc0","passage_index":0,"mention":"mesothelial cell","start":4,"end":20,"entity_type":"cell_type","cl_id":"CL:0000077","cl_label":"mesothelial cell","score":0.991,"source":"dense_retrieval"}
```

Fields:

| Field | Type | Description |
|---|---|---|
| `document_id` | string | Document identifier. |
| `passage_index` | integer | Passage index within the document. |
| `mention` | string | Detected cell-type mention. |
| `start` | integer or null | Absolute start offset. |
| `end` | integer or null | Absolute end offset. |
| `entity_type` | string or null | Entity label, usually `cell_type`. |
| `cl_id` | string or null | Predicted Cell Ontology ID. |
| `cl_label` | string or null | Linked ontology label or alias. |
| `score` | number or null | Linker score. |
| `source` | string or null | Matching source. |

---

## 8. Passage JSONL format

The shared I/O utilities use a passage-level JSONL format.

Example:

```json
{
  "id": 0,
  "document_id": "sample-doc-001",
  "passage_id": 0,
  "passage_offset": 0,
  "text": "The mesothelial cell and SMC clusters formed the third population.",
  "entities": [
    {
      "start": 4,
      "end": 20,
      "label": "cell_type",
      "text": "mesothelial cell"
    }
  ]
}
```

`start` and `end` are passage-relative offsets in this JSONL representation unless otherwise specified by the calling function. BioC XML output converts them to absolute offsets using `passage_offset`.

---

## 9. Ontology JSONL resource format

The Cell Ontology resource should be JSONL, with one concept per line.

Recommended fields:

```json
{
  "id": "CL:0000077",
  "label": "mesothelial cell",
  "synonyms": ["mesotheliocyte"],
  "definition": "..."
}
```

Accepted alternative field names may include:

```text
cl_id
identifier
curie
name
preferred_label
aliases
exact_synonyms
related_synonyms
```

The ontology loader is designed to be permissive, but for reproducibility you should keep one stable resource file in:

```text
src/cellexlink/resources/cell_ontology_v2025-12-17.jsonl
```

---

## 10. Abbreviation TSV format

The abbreviation dictionary should be a tab-separated file.

Recommended columns:

```text
short_form	cl_id	label
SMC	CL:0000192	smooth muscle cell
```

If a short form maps to more than one CL ID, it is treated as ambiguous. Ambiguous abbreviations can be resolved using document-level long-form recovery when context is available.

Example:

```text
EC	CL:0000115	endothelial cell
EC	CL:0000066	epithelial cell
```

---

## 11. Runtime summary files

Prediction modules may write runtime summaries such as:

```text
predict_runtime_summary.json
normalization_runtime_summary.json
```

These files are useful for benchmarking and debugging but are not part of the required output format.

---

## 12. Recommended file naming

For a corpus named `my_corpus`, use:

```text
outputs/my_corpus/
├── ner_predictions.xml
├── normalized.xml
├── predictions.jsonl
├── ner_work/
└── normalization_work/
```

This makes benchmarking and debugging easier because intermediate and final outputs are clearly separated.

