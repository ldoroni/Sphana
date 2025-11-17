"""ONNX export helpers."""

from __future__ import annotations

import logging
import warnings
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification

from sphana_trainer.models import EmbeddingEncoder, GGNNRanker


LOGGER = logging.getLogger(__name__)


def export_embedding_encoder(
    model: EmbeddingEncoder,
    tokenizer,
    output_dir: Path,
    opset: int,
    max_seq_length: int,
    quantize: bool,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "embedding.onnx"
    model_cpu = model.to("cpu").eval()
    dummy = tokenizer(
        ["export sample"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=torch.jit.TracerWarning)
        torch.onnx.export(
            model_cpu,
            (dummy["input_ids"], dummy["attention_mask"]),
            onnx_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["embeddings"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "embeddings": {0: "batch"},
            },
            opset_version=opset,
            dynamo=False,
        )

    quant_path = (
        _quantize_or_fallback(onnx_path, output_dir / "embedding.int8.onnx")
        if quantize
        else onnx_path
    )
    torch_reference = (
        model_cpu(dummy["input_ids"], dummy["attention_mask"]).detach().cpu().numpy()
    )
    feeds = {
        "input_ids": dummy["input_ids"].cpu().numpy(),
        "attention_mask": dummy["attention_mask"].cpu().numpy(),
    }
    _validate_onnx_model(onnx_path, feeds, reference=torch_reference)
    if quant_path != onnx_path:
        _validate_onnx_model(quant_path, feeds, reference=torch_reference, atol=5e-2, rtol=2e-1)
    return onnx_path, quant_path


def export_relation_classifier(
    model_dir: Path,
    tokenizer,
    output_dir: Path,
    opset: int,
    max_seq_length: int,
    quantize: bool,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "relation.onnx"
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to("cpu").eval()
    dummy = tokenizer(
        ["export sample"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
    )
    inputs = [dummy["input_ids"], dummy["attention_mask"]]
    input_names = ["input_ids", "attention_mask"]
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "logits": {0: "batch"},
    }
    if "token_type_ids" in dummy:
        inputs.append(dummy["token_type_ids"])
        input_names.append("token_type_ids")
        dynamic_axes["token_type_ids"] = {0: "batch", 1: "sequence"}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=torch.jit.TracerWarning)
        torch.onnx.export(
            model,
            tuple(inputs),
            onnx_path,
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            dynamo=False,
        )
    quant_path = (
        _quantize_or_fallback(onnx_path, output_dir / "relation.int8.onnx")
        if quantize
        else onnx_path
    )
    torch_reference = model(**dummy).logits.detach().cpu().numpy()
    feeds = {
        "input_ids": dummy["input_ids"].cpu().numpy(),
        "attention_mask": dummy["attention_mask"].cpu().numpy(),
    }
    if "token_type_ids" in dummy:
        feeds["token_type_ids"] = dummy["token_type_ids"].cpu().numpy()
    _validate_onnx_model(onnx_path, feeds, reference=torch_reference)
    if quant_path != onnx_path:
        _validate_onnx_model(quant_path, feeds, reference=torch_reference, atol=5e-2, rtol=2e-1)
    return onnx_path, quant_path


def export_gnn_ranker(
    model: GGNNRanker,
    output_dir: Path,
    opset: int,
    max_nodes: int,
    max_edges: int,
    quantize: bool,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "gnn.onnx"
    model_cpu = model.to("cpu").eval()
    dummy_nodes = torch.zeros(max_nodes, model.input_dim, dtype=torch.float32)
    dummy_edge_index = torch.zeros(max_edges, 2, dtype=torch.long)
    dummy_edge_dirs = torch.zeros(max_edges, dtype=torch.long)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        warnings.simplefilter("ignore", category=torch.jit.TracerWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        torch.onnx.export(
            model_cpu,
            (dummy_nodes, dummy_edge_index, dummy_edge_dirs),
            onnx_path,
            input_names=["node_features", "edge_index", "edge_directions"],
            output_names=["score"],
            dynamic_axes={
                "node_features": {0: "nodes"},
                "edge_index": {0: "edges"},
                "edge_directions": {0: "edges"},
            },
            opset_version=opset,
            dynamo=False,
        )

    quant_path: Optional[Path] = None
    if quantize:
        quant_path = _quantize_or_fallback(onnx_path, output_dir / "gnn.int8.onnx")
    feeds = {
        "node_features": dummy_nodes.numpy(),
        "edge_index": dummy_edge_index.numpy(),
        "edge_directions": dummy_edge_dirs.numpy(),
    }
    _validate_onnx_model(onnx_path, feeds)
    if quant_path:
        _validate_onnx_model(quant_path, feeds)
    return onnx_path, quant_path or onnx_path


def _quantize_or_fallback(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    preprocessed = _maybe_preprocess_model(source)
    quant_input = preprocessed if preprocessed.exists() else source
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            with _suppress_quant_warning():
                quantize_dynamic(
                    model_input=str(quant_input),
                    model_output=str(target),
                    weight_type=QuantType.QInt8,
                )
        return target
    except Exception as exc:  # pragma: no cover - hardware/env specific
        LOGGER.warning("Quantization failed for %s: %s. Using original ONNX.", source, exc)
        if target.exists():
            target.unlink(missing_ok=True)  # type: ignore[attr-defined]
        return source
    finally:
        if preprocessed != source:
            with suppress(FileNotFoundError):
                preprocessed.unlink()


def _maybe_preprocess_model(source: Path) -> Path:
    try:
        from onnxruntime.quantization.preprocess import quant_preprocess
    except ImportError:  # pragma: no cover - optional dependency inside onnxruntime
        return source

    preprocessed = source.with_suffix(".pre.onnx")
    try:
        quant_preprocess(
            input_model_path=str(source),
            output_model_path=str(preprocessed),
            skip_model_check=True,
            auto_merge=True,
            optimize_model=True,
            float16=False,
        )
        return preprocessed
    except Exception as exc:  # pragma: no cover - preprocess best-effort
        LOGGER.debug("Quantization preprocess skipped for %s: %s", source, exc)
        with suppress(FileNotFoundError):
            preprocessed.unlink()
        return source


@contextmanager
def _suppress_quant_warning():
    message = "Please consider to run pre-processing before quantization."
    root_logger = logging.getLogger()

    class _Filter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
            return message not in record.getMessage()

    filt = _Filter()
    root_logger.addFilter(filt)
    try:
        yield
    finally:
        root_logger.removeFilter(filt)


def _validate_onnx_model(
    onnx_path: Path,
    feeds: Dict[str, np.ndarray],
    reference: Optional[np.ndarray] = None,
    *,
    atol: float = 1e-3,
    rtol: float = 1e-2,
) -> None:
    try:
        session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:  # pragma: no cover - environment specific
        raise RuntimeError(f"Failed to initialize ONNX Runtime session for {onnx_path}: {exc}") from exc

    try:
        outputs = session.run(None, feeds)
    except Exception as exc:  # pragma: no cover - environment specific
        raise RuntimeError(f"ONNX Runtime inference failed for {onnx_path}: {exc}") from exc
    if reference is not None:
        ort_output = outputs[0]
        if ort_output.shape != reference.shape:
            raise RuntimeError(
                f"ONNX output shape {ort_output.shape} does not match PyTorch reference {reference.shape} for {onnx_path}."
            )
        np.testing.assert_allclose(ort_output, reference, atol=atol, rtol=rtol)


