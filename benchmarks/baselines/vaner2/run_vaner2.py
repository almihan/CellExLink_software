#!/usr/bin/env python3
"""Run VANER2 NER on CellExLink BioC XML benchmark files.

Place this file at:
    benchmarks/baselines/vaner2/run_vaner2.py

Expected VANER2 assets under the same directory:
    model/
    finetuned_models/<model_name>/saved.pt
    LLAMA3.1_prompt.txt   # only needed for Llama prompt models

Default input discovery:
    benchmarks/data/evaluation/*/test.xml

Default output:
    benchmarks/baselines/vaner2/model_outputs/<data_filename>.xml
"""
from __future__ import annotations

import argparse
import copy
import csv
import glob
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

import numpy as np
import torch
import torch._dynamo
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

BASELINE_DIR = Path(__file__).resolve().parent
BENCHMARKS_DIR = BASELINE_DIR.parents[1]
DEFAULT_DATA_ROOT = BENCHMARKS_DIR / "data" / "evaluation"
DEFAULT_OUTPUT_DIR = BASELINE_DIR / "model_outputs"
DEFAULT_MODEL_ROOT = BASELINE_DIR / "finetuned_models"
DEFAULT_MODEL_NAME = "VANER2"
DEFAULT_DATASET_FOLDER = "CellLinkXML"
VANER2_ENTITY_TYPE = "CellType"
OUTPUT_ENTITY_TYPE = "cell_type"

# Match VANER2's original relative-import/relative-file behavior.
os.chdir(BASELINE_DIR)
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

torch._dynamo.config.suppress_errors = True

from model.data import read_data  # noqa: E402
from model.models import MLP_Head, UnmaskingLlamaForTokenClassification  # noqa: E402
from model.utils import (  # noqa: E402
    bio2brat,
    get_bert_base_params,
    get_llm_base_params,
    get_special_token_ids,
    post_process,
    save_predictions,
)

PassageInfo = Tuple[ET.Element, ET.Element, int]


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def local_name(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def direct_child(parent: ET.Element, name: str) -> Optional[ET.Element]:
    return next((child for child in parent if local_name(child.tag) == name), None)


def direct_children(parent: ET.Element, name: str) -> Iterable[ET.Element]:
    return (child for child in parent if local_name(child.tag) == name)


def child_text(parent: ET.Element, name: str) -> str:
    child = direct_child(parent, name)
    return "" if child is None or child.text is None else child.text


def next_annotation_id(document: ET.Element) -> int:
    max_id = -1
    for annotation in document.iter():
        if local_name(annotation.tag) != "annotation":
            continue
        raw_id = annotation.get("id")
        if raw_id is None:
            continue
        try:
            max_id = max(max_id, int(raw_id))
        except ValueError:
            continue
    return max_id + 1


def clear_passage_predictions(passage: ET.Element) -> None:
    for child in list(passage):
        if local_name(child.tag) in {"annotation", "relation"}:
            passage.remove(child)


def append_annotation(
    passage: ET.Element,
    annotation_id: int,
    offset: int,
    length: int,
    text: str,
) -> None:
    annotation = ET.SubElement(passage, "annotation", {"id": str(annotation_id)})
    infon = ET.SubElement(annotation, "infon", {"key": "type"})
    infon.text = OUTPUT_ENTITY_TYPE
    ET.SubElement(annotation, "location", {"offset": str(offset), "length": str(length)})
    text_node = ET.SubElement(annotation, "text")
    text_node.text = text


def load_xml_and_write_pubtator(
    input_xml: Path,
    output_pubtator: Path,
) -> Tuple[ET.ElementTree, Dict[str, PassageInfo]]:
    tree = ET.parse(input_xml)
    root = tree.getroot()
    output_pubtator.parent.mkdir(parents=True, exist_ok=True)
    passage_lookup: Dict[str, PassageInfo] = {}

    with output_pubtator.open("w", encoding="utf-8") as handle:
        for doc_index, document in enumerate(direct_children(root, "document"), start=1):
            document_id = child_text(document, "id").strip() or f"doc-{doc_index}"
            for passage_index, passage in enumerate(direct_children(document, "passage"), start=1):
                clear_passage_predictions(passage)
                text_node = direct_child(passage, "text")
                offset_text = child_text(passage, "offset").strip()
                if text_node is None or text_node.text is None or not offset_text:
                    continue
                passage_text = text_node.text
                if not passage_text.strip():
                    continue

                passage_id = f"{document_id}__p{passage_index}"
                passage_lookup[passage_id] = (document, passage, int(offset_text))
                handle.write(f"{passage_id}|t|{passage_text}\n")
                handle.write(f"{passage_id}|a|\n\n")

    return tree, passage_lookup


@torch.no_grad()
def batch_inference(encoder, type_heads, data, tokenizer, eval_params):
    tokens = data["tokens"]
    start_pos = data["token_start_pos"]

    if not eval_params["use_prompt"]:
        prompt = "."
    elif eval_params["model_type"] == "bert":
        prompt = "Extract all biomedical entities from the following text: "
    else:
        prompt = (BASELINE_DIR / "LLAMA3.1_prompt.txt").read_text(encoding="utf-8")

    prompt_tokens = tokenizer(prompt, add_special_tokens=False).data["input_ids"]
    prompt_tokens = torch.tensor(prompt_tokens)
    real_len = eval_params["cxt_len"] - prompt_tokens.size(0)
    if real_len <= 0:
        raise ValueError("Prompt length is greater than or equal to the model context length.")
    step_size = real_len // 2

    concated_token = []
    concated_start_pos = []
    concated_doc_id = []
    for doc_id in range(len(tokens)):
        concated_token += tokens[doc_id] + [eval_params["sep_id"]]
        concated_start_pos += start_pos[doc_id]
        concated_doc_id += [doc_id for _ in start_pos[doc_id]]

    input_ids = torch.tensor(concated_token)
    batch = []
    pos = 0
    while pos + real_len < input_ids.size(0):
        batch.append(torch.cat((prompt_tokens, input_ids[pos : pos + real_len]), 0))
        pos += step_size

    temp = input_ids[pos:]
    last = torch.zeros(real_len, dtype=input_ids.dtype) + eval_params["pad_id"]
    last[: temp.size(0)] = temp
    batch.append(torch.cat((prompt_tokens, last), 0))
    batch = torch.stack(batch, 0).cuda()

    sbatch_size = 16
    sbatches = [batch[i : i + sbatch_size] for i in range(0, batch.size(0), sbatch_size)]

    all_type_best_tags = {entity_type: [] for entity_type in type_heads}
    all_type_probs = {entity_type: [] for entity_type in type_heads}
    for sbatch in tqdm(sbatches, total=len(sbatches)):
        mask = torch.ones_like(sbatch)
        mask[sbatch == 0] = 0
        output_embed = encoder(sbatch, attention_mask=mask).last_hidden_state
        for entity_type in type_heads:
            _, best_tags, probs = type_heads[entity_type](output_embed)
            all_type_best_tags[entity_type].append(best_tags[:, -real_len:])
            all_type_probs[entity_type].append(probs[:, -real_len:])

    all_type_all_doc_predictions = {}
    texts = [
        raw_text[0].split("|t|")[1] + raw_text[1].split("|a|")[1].strip("\n")
        for raw_text in data["raw_text"]
    ]

    for entity_type in type_heads:
        best_tags = torch.cat(all_type_best_tags[entity_type], 0)
        probs = torch.cat(all_type_probs[entity_type], 0)

        pos2 = copy.deepcopy(pos)
        pred_tags = torch.zeros(input_ids.size(0))
        pred_probs = torch.zeros(input_ids.size(0))
        half = (real_len - step_size) // 2
        rest_len = pred_tags.size(0) - pos2

        for j in range(best_tags.size(0)):
            source_idx = best_tags.size(0) - j - 1
            if j == 0:
                pred_tags[pos2:] = best_tags[source_idx, :rest_len].cpu()
                pred_probs[pos2:] = probs[source_idx, :rest_len].cpu()
            else:
                pred_tags[pos2 : pos2 + real_len - half] = best_tags[source_idx, :-half].cpu()
                pred_probs[pos2 : pos2 + real_len - half] = probs[source_idx, :-half].cpu()
            pos2 -= step_size

        try:
            import NER_helper_functions

            all_doc_predictions = NER_helper_functions.bio2brat(
                list(pred_tags.cpu().numpy().astype(np.int32)),
                list(pred_probs.cpu().numpy().astype(np.float64)),
                concated_doc_id,
                concated_start_pos,
                [len(text) for text in texts],
                eval_params["use_bioe"],
            )
        except ImportError:
            print("Using python for bio2brat")
            all_doc_tags = {}
            all_doc_predictions = {}
            for tag, prob, doc_id, start in zip(pred_tags, pred_probs, concated_doc_id, concated_start_pos):
                all_doc_tags.setdefault(doc_id, []).append((tag, prob, start))
            for doc_id in all_doc_tags:
                all_doc_tags[doc_id].append((0, 1, len(texts[doc_id])))
                all_doc_predictions[doc_id] = bio2brat(all_doc_tags[doc_id], use_bioe=eval_params["use_bioe"])

        all_type_all_doc_predictions[entity_type] = all_doc_predictions

    return all_type_all_doc_predictions


@torch.no_grad()
def save_vaner2_results(test_data, all_model_predictions, use_score_threshold: bool, save_path: Path) -> None:
    score_threshold = (len(all_model_predictions) + 1) * 0.35 if use_score_threshold else 0
    merged = all_model_predictions[0]

    for data in test_data:
        for model in all_model_predictions[1:]:
            for entity_type in model[data["dataset"]]:
                for doc_id in range(len(model[data["dataset"]][entity_type])):
                    merged[data["dataset"]][entity_type][doc_id] += model[data["dataset"]][entity_type][doc_id]

    for data in test_data:
        for entity_type in merged[data["dataset"]]:
            for doc_id in range(len(merged[data["dataset"]][entity_type])):
                merged[data["dataset"]][entity_type][doc_id] = post_process(
                    data["raw_text"][doc_id],
                    merged[data["dataset"]][entity_type][doc_id],
                    data["dataset"],
                    score_threshold,
                )

    predictions_dir = save_path / "predictions"
    if predictions_dir.exists():
        shutil.rmtree(predictions_dir)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    for data in test_data:
        name = data["dataset"] + ".pubtator"
        save_predictions(
            data["raw_text"],
            merged[data["dataset"]],
            str(predictions_dir / name),
            with_score=True,
        )


@torch.no_grad()
def run_single_model(model_path: Path, dataset_folder: str):
    print(f"Evaluating model: {model_path}")
    saved_path = model_path / "saved.pt"
    if not saved_path.exists():
        raise FileNotFoundError(f"Missing VANER2 checkpoint: {saved_path}")

    saved = torch.load(str(saved_path), weights_only=True)
    model_config = saved["train_config"] if "train_config" in saved else saved["model_config"]
    base_model_path = saved["base_model_path"]
    model_type = "llm" if "Llama" in base_model_path else "bert"
    use_bioe = model_config["num_classes"] == 4
    use_prompt = model_config["use_prompt"]

    label2id = {"O": 0, "I": 1, "B": 2, "E": 3} if use_bioe else {"O": 0, "I": 1, "B": 2}
    id2label = {value: key for key, value in label2id.items()}
    test_path = BASELINE_DIR / "data" / dataset_folder / "test"

    print("Evaluating PubTator files:")
    for path in glob.glob(str(test_path / "**" / "*.pubtator"), recursive=True):
        print(f"  {path}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if model_type == "llm":
        encoder = UnmaskingLlamaForTokenClassification.from_pretrained(
            base_model_path,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
        ).bfloat16().to("cuda")
        pretrain_path = model_path / "pretrain"
        if pretrain_path.exists():
            encoder = PeftModel.from_pretrained(encoder, str(pretrain_path))
            encoder = encoder.merge_and_unload()
        encoder = PeftModel.from_pretrained(encoder, str(model_path))
        cxt_len, embed_size, _, _ = get_llm_base_params(base_model_path.split("/")[-1])
    elif model_type == "bert":
        encoder = AutoModel.from_pretrained(str(model_path)).bfloat16().cuda()
        cxt_len, embed_size, _, _ = get_bert_base_params(base_model_path.split("/")[-1])
    else:
        raise TypeError(f"Unknown VANER2 model type: {model_type}")

    encoder.eval()
    torch.set_float32_matmul_precision("high")
    encoder = torch.compile(encoder)

    type_heads = {}
    for entity_type, state_dict in saved["type_heads_state_dict"].items():
        head = MLP_Head(model_config, embed_size).bfloat16().cuda()
        head.load_state_dict(state_dict)
        type_heads[entity_type] = torch.compile(head)

    pad_id, sep_id, _, _ = get_special_token_ids(
        tokenizer,
        model_type,
        base_model_path.split("/")[-1],
    )
    test_data = read_data(
        str(test_path) + "/",
        mode="eval",
        tokenizer=tokenizer,
        label2id=label2id,
        use_bioe=use_bioe,
        verbose=True,
    )
    eval_params = {
        "model_type": model_type,
        "use_prompt": use_prompt,
        "use_bioe": use_bioe,
        "cxt_len": cxt_len,
        "pad_id": pad_id,
        "sep_id": sep_id,
    }

    predictions = {}
    for data in test_data:
        predictions[data["dataset"]] = batch_inference(encoder, type_heads, data, tokenizer, eval_params)
    return test_data, predictions


def run_vaner2(
    dataset_folder: str,
    model_names: Sequence[str],
    model_root: Path,
    merge_all_model_outputs: bool,
    use_score_threshold: bool,
) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("VANER2's original runner requires a CUDA GPU.")

    if merge_all_model_outputs:
        test_data = None
        all_model_predictions = []
        for model_name in model_names:
            this_test_data, predictions = run_single_model(model_root / model_name, dataset_folder)
            test_data = this_test_data
            all_model_predictions.append(predictions)
        if test_data is None or not all_model_predictions:
            raise RuntimeError("No VANER2 models were evaluated.")
        save_path = BASELINE_DIR / "results" / dataset_folder / "merged"
        save_vaner2_results(test_data, all_model_predictions, use_score_threshold, save_path)
    else:
        if len(model_names) != 1:
            raise ValueError("Use exactly one --model_names value, or set --merge_all_model_outputs true.")
        test_data, predictions = run_single_model(model_root / model_names[0], dataset_folder)
        save_path = BASELINE_DIR / "results" / dataset_folder / model_names[0]
        save_vaner2_results(test_data, [predictions], use_score_threshold, save_path)

    result_pubtator = save_path / "predictions" / f"{dataset_folder}.pubtator"
    if not result_pubtator.exists():
        raise FileNotFoundError(f"Prediction file not found: {result_pubtator}")
    return result_pubtator


def apply_predictions_to_xml(
    tree: ET.ElementTree,
    passage_lookup: Dict[str, PassageInfo],
    result_pubtator: Path,
    output_xml: Path,
) -> int:
    doc_next_ids: Dict[int, int] = {}
    annotations_added = 0

    with result_pubtator.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line or "|t|" in line or "|a|" in line:
                continue
            fields = line.split("\t")
            if len(fields) < 5:
                continue
            passage_id, start, end, mention_text, entity_type = fields[:5]
            if entity_type != VANER2_ENTITY_TYPE or passage_id not in passage_lookup:
                continue
            try:
                start_i = int(start)
                end_i = int(end)
            except ValueError:
                continue

            document, passage, passage_offset = passage_lookup[passage_id]
            passage_text = child_text(passage, "text")
            if start_i < 0 or end_i <= start_i or end_i > len(passage_text):
                continue

            doc_key = id(document)
            if doc_key not in doc_next_ids:
                doc_next_ids[doc_key] = next_annotation_id(document)

            span_text = passage_text[start_i:end_i]
            append_annotation(
                passage=passage,
                annotation_id=doc_next_ids[doc_key],
                offset=passage_offset + start_i,
                length=end_i - start_i,
                text=span_text or mention_text,
            )
            doc_next_ids[doc_key] += 1
            annotations_added += 1

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    return annotations_added


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return value or "dataset"


def discover_inputs(data_root: Path) -> List[Path]:
    preferred = sorted(data_root.glob("*/test.xml"))
    return preferred if preferred else sorted(data_root.glob("**/*.xml"))


def dataset_folder_for_input(input_xml: Path, total: int, explicit: Optional[str]) -> str:
    if explicit:
        if total > 1:
            raise ValueError("--dataset_folder can only be used with one --input_xml file.")
        return explicit
    if total == 1:
        return DEFAULT_DATASET_FOLDER
    return sanitize_name(f"{input_xml.parent.name}_{input_xml.stem}")


def output_path_for_input(input_xml: Path, output_dir: Path, total: int) -> Path:
    if total == 1:
        return output_dir / input_xml.name
    return output_dir / f"{sanitize_name(input_xml.parent.name)}_{input_xml.name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VANER2 on CellExLink BioC XML benchmark files.")
    parser.add_argument(
        "--input_xml",
        "--input-xml",
        dest="input_xmls",
        action="append",
        type=Path,
        help="Input BioC XML file. Repeat for multiple files. Default: discover */test.xml under --data_root.",
    )
    parser.add_argument(
        "--data_root",
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Benchmark data root used when --input_xml is omitted. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--model_root",
        "--model-root",
        type=Path,
        default=DEFAULT_MODEL_ROOT,
        help=f"Directory containing VANER2 fine-tuned models. Default: {DEFAULT_MODEL_ROOT}",
    )
    parser.add_argument(
        "--model_names",
        "--model-names",
        nargs="+",
        default=[DEFAULT_MODEL_NAME],
        help=f"Fine-tuned model name(s) under --model_root. Default: {DEFAULT_MODEL_NAME}",
    )
    parser.add_argument(
        "--dataset_folder",
        "--dataset-folder",
        default=None,
        help=(
            "Temporary VANER2 dataset folder under baselines/vaner2/data/. "
            f"Default for one input: {DEFAULT_DATASET_FOLDER}; for multiple inputs: <dataset>_<filename>."
        ),
    )
    parser.add_argument(
        "--merge_all_model_outputs",
        "--merge-all-model-outputs",
        type=str_to_bool,
        default=False,
        help="Merge predictions from all --model_names values by VANER2 voting. Default: false.",
    )
    parser.add_argument(
        "--use_score_threshold",
        "--use-score-threshold",
        type=str_to_bool,
        default=True,
        help="Use VANER2's score threshold during post-processing. Default: true.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_xmls = [path.resolve() for path in (args.input_xmls or discover_inputs(args.data_root.resolve()))]
    if not input_xmls:
        raise FileNotFoundError(f"No XML files found. Provide --input_xml or check --data_root: {args.data_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    total = len(input_xmls)

    for input_xml in input_xmls:
        if not input_xml.exists():
            raise FileNotFoundError(f"Input XML not found: {input_xml}")

        dataset_folder = dataset_folder_for_input(input_xml, total, args.dataset_folder)
        input_pubtator = BASELINE_DIR / "data" / dataset_folder / "test" / f"{dataset_folder}.pubtator"
        output_xml = output_path_for_input(input_xml, args.output_dir, total)

        print(f"\nInput XML:        {input_xml}")
        print(f"Temp PubTator:    {input_pubtator}")
        print(f"Dataset folder:   {dataset_folder}")
        print(f"Output XML:       {output_xml}")

        tree, passage_lookup = load_xml_and_write_pubtator(input_xml, input_pubtator)
        print(f"Passages written: {len(passage_lookup)}")

        result_pubtator = run_vaner2(
            dataset_folder=dataset_folder,
            model_names=args.model_names,
            model_root=args.model_root.resolve(),
            merge_all_model_outputs=args.merge_all_model_outputs,
            use_score_threshold=args.use_score_threshold,
        )
        annotations_added = apply_predictions_to_xml(tree, passage_lookup, result_pubtator, output_xml)
        print(f"Annotations added: {annotations_added}")

        manifest_rows.append(
            {
                "input_xml": str(input_xml),
                "dataset_folder": dataset_folder,
                "input_pubtator": str(input_pubtator),
                "prediction_pubtator": str(result_pubtator),
                "output_xml": str(output_xml),
                "annotations_added": annotations_added,
            }
        )

    manifest_path = args.output_dir / "vaner2_run_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\nManifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
