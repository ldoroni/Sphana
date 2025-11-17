"""Generate parity samples for ONNX inference to assist .NET validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from sphana_trainer.utils.metadata import load_artifact_metadata
from sphana_trainer.data.relation_dataset import _insert_entity_markers


def generate_parity_sample(component: str, sample_file: Path, artifact_root: Path) -> Dict[str, Any]:
    component = component.lower()
    if component not in {"embedding", "relation", "gnn"}:
        raise ValueError("Component must be embedding, relation, or gnn.")

    meta = load_artifact_metadata(component, artifact_root)
    onnx_path = Path(meta.quantized_path) if meta.quantized_path else Path(meta.onnx_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    sample = _load_first_record(sample_file)

    if component == "embedding":
        payload = _embedding_parity(meta, sample, session)
    elif component == "relation":
        payload = _relation_parity(meta, sample, session)
    else:
        payload = _gnn_parity(sample, session)

    return {
        "component": component,
        "version": meta.version,
        "onnx_path": meta.onnx_path,
        "inputs": payload["inputs"],
        "outputs": payload["outputs"],
    }


def _embedding_parity(meta, sample, session):
    tokenizer = AutoTokenizer.from_pretrained(meta.config["model_name"], use_fast=True)
    text = sample.get("query") or sample.get("anchor") or sample.get("text")
    if not text:
        raise ValueError("Embedding sample must include 'query' or 'anchor' text.")
    encoded = tokenizer(
        text,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=int(meta.config.get("max_seq_length", 128)),
    )
    session_inputs = {inp.name for inp in session.get_inputs()} if hasattr(session, "get_inputs") else set()
    feeds = {
        "input_ids": encoded["input_ids"].astype("int64"),
        "attention_mask": encoded["attention_mask"].astype("int64"),
    }
    if "token_type_ids" in encoded and "token_type_ids" in session_inputs:
        feeds["token_type_ids"] = encoded["token_type_ids"].astype("int64")
    outputs = session.run(None, feeds)
    return {
        "inputs": {
            "text": text,
            "input_ids": feeds["input_ids"][0].tolist(),
            "attention_mask": feeds["attention_mask"][0].tolist(),
        },
        "outputs": {"embedding": outputs[0][0].tolist()},
    }


def _relation_parity(meta, sample, session):
    tokenizer = AutoTokenizer.from_pretrained(meta.config["model_name"], use_fast=True)
    text = sample.get("text")
    entity1 = sample.get("entity1")
    entity2 = sample.get("entity2")
    if not (text and entity1 and entity2):
        raise ValueError("Relation sample must include text, entity1, and entity2.")
    marked = _insert_entity_markers(text, entity1, entity2)
    encoded = tokenizer(
        marked,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=int(meta.config.get("max_seq_length", 256)),
    )
    session_inputs = {inp.name for inp in session.get_inputs()} if hasattr(session, "get_inputs") else set()
    feeds = {
        "input_ids": encoded["input_ids"].astype("int64"),
        "attention_mask": encoded["attention_mask"].astype("int64"),
    }
    if "token_type_ids" in encoded and "token_type_ids" in session_inputs:
        feeds["token_type_ids"] = encoded["token_type_ids"].astype("int64")
    outputs = session.run(None, feeds)
    logits = outputs[0][0]
    probs = _softmax(logits)
    inputs = {
        "text": text,
        "marked_text": marked,
        "entity1": entity1,
        "entity2": entity2,
        "input_ids": feeds["input_ids"][0].tolist(),
        "attention_mask": feeds["attention_mask"][0].tolist(),
    }
    if "token_type_ids" in feeds:
        inputs["token_type_ids"] = feeds["token_type_ids"][0].tolist()
    return {
        "inputs": inputs,
        "outputs": {"logits": logits.tolist(), "probabilities": probs.tolist()},
    }


def _gnn_parity(sample, session):
    if "candidates" not in sample or not sample["candidates"]:
        raise ValueError("GNN sample must include at least one candidate.")
    candidate = sample["candidates"][0]
    node_features = np.array(candidate["node_features"], dtype=np.float32)
    edge_index = np.array(candidate.get("edge_index", []), dtype=np.int64)
    edge_dirs = np.array(candidate.get("edge_directions", []), dtype=np.int64)
    if edge_index.shape == (0,):
        edge_index = np.zeros((0, 2), dtype=np.int64)
    feeds = {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_directions": edge_dirs,
    }
    outputs = session.run(None, feeds)
    score = float(np.asarray(outputs[0]).reshape(-1)[0])
    return {
        "inputs": {
            "query": sample.get("query_id"),
            "candidate": candidate,
        },
        "outputs": {"score": score},
    }


def _load_first_record(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError(f"No records found in {path}")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()

