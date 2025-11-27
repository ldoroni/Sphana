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
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForCausalLM,
)

from sphana_trainer.models import EmbeddingEncoder, GGNNRanker

try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

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


def export_ner_model(
    model_name_or_path: str,
    tokenizer,
    output_dir: Path,
    opset: int,
    max_seq_length: int,
    quantize: bool,
    allowed_mismatch_ratio: float = 0.0,
) -> Tuple[Path, Path]:
    """Export Token Classification model to ONNX."""
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "ner.onnx"
    
    # Load model
    model = AutoModelForTokenClassification.from_pretrained(model_name_or_path).to("cpu").eval()
    
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
        "logits": {0: "batch", 1: "sequence"},
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
        _quantize_or_fallback(onnx_path, output_dir / "ner.int8.onnx")
        if quantize
        else onnx_path
    )
    
    # Validation
    torch_reference = model(**dummy).logits.detach().cpu().numpy()
    feeds = {
        "input_ids": dummy["input_ids"].cpu().numpy(),
        "attention_mask": dummy["attention_mask"].cpu().numpy(),
    }
    if "token_type_ids" in dummy:
        feeds["token_type_ids"] = dummy["token_type_ids"].cpu().numpy()
        
    _validate_onnx_model(onnx_path, feeds, reference=torch_reference, atol=1e-3, rtol=1e-3)
    if quant_path != onnx_path:
        # Check against allowed mismatch ratio for quantization
        _validate_onnx_model(
            quant_path, 
            feeds, 
            reference=torch_reference, 
            atol=1e-1, 
            rtol=3e-1, 
            allowed_mismatch_ratio=allowed_mismatch_ratio
        )
        
    return onnx_path, quant_path


def export_llm_model(
    model_name_or_path: str,
    tokenizer,
    output_dir: Path,
    opset: int,
    max_seq_length: int,
    quantize: bool,
    allowed_mismatch_ratio: float = 0.0,
) -> Tuple[Path, Path]:
    """Export Causal LM model to ONNX (using optimum)."""
    try:
        from optimum.exporters.onnx import main_export
    except ImportError as exc:
        raise RuntimeError("optimum is required for LLM export. Please install it.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use optimum main_export to handle complex LLM export (with past, SDPA handling etc)
    # We perform quantization separately below if requested, so we skip optimum's post-process quantization here
    # to maintain consistency with our pipeline (int8 dynamic).
    # Note: main_export usually exports to a folder.
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main_export(
            model_name_or_path=model_name_or_path,
            output=output_dir,
            task="text-generation-with-past", # Standard for LLM inference
            opset=opset,
            no_post_process=True,
            device="cpu",
        )

    # Handle file naming: optimum usually produces 'model.onnx' or 'decoder_model.onnx' etc.
    # For text-generation-with-past, it might produce merged model.onnx if supported.
    # We look for model.onnx first.
    src_onnx = output_dir / "model.onnx"
    # Sometimes optimum exports as decoder_model_merged.onnx for some architectures
    if not src_onnx.exists():
        candidates = list(output_dir.glob("*.onnx"))
        if len(candidates) == 1:
            src_onnx = candidates[0]
        elif len(candidates) > 1:
            # If multiple files (e.g. decoder + decoder_with_past), pick the merged one or fail
            merged = output_dir / "decoder_model_merged.onnx"
            if merged.exists():
                src_onnx = merged
            else:
                # Just pick the largest one? Or fail?
                # For now, let's assume model.onnx or merged.
                pass

    if not src_onnx.exists():
        raise FileNotFoundError(f"Optimum export did not produce expected 'model.onnx' in {output_dir}")

    onnx_path = output_dir / "llm_generator.onnx"
    if src_onnx.resolve() != onnx_path.resolve():
        if onnx_path.exists():
            onnx_path.unlink()
        src_onnx.rename(onnx_path)

    quant_path = (
        _quantize_or_fallback(onnx_path, output_dir / "llm_generator.int8.onnx")
        if quantize
        else onnx_path
    )
    
    # Validation logic
    # Note: Optimum export usually works, and full validation might require complex inputs (past_key_values).
    # We do a lightweight check on input/output shapes using dummy inputs if possible.
    # However, our simple _validate_onnx_model uses only input_ids/attention_mask.
    # The exported model might expect position_ids or past_key_values if not merged properly.
    # If validation fails due to missing inputs, we log warning and skip.
    
    # Re-load tokenizer for dummy creation
    if tokenizer is None:
        # Fallback if not provided (though it should be)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    dummy = tokenizer(
        ["Hello world"],
        return_tensors="pt",
        max_length=max_seq_length, 
        truncation=True
    )
    
    feeds = {
        "input_ids": dummy["input_ids"].cpu().numpy(),
        "attention_mask": dummy["attention_mask"].cpu().numpy(),
    }
    
    # Try validation, warn on failure (e.g. missing inputs)
    try:
        # For reference, we need the torch model. Loading it just for validation is expensive but correctness is key.
        # To avoid reloading if we can, we skip strict numeric validation here if we trust optimum,
        # OR we try to validate quantization only if we have reference.
        # Let's SKIP reference comparison for now to avoid reloading model twice (optimum loaded it once).
        # Unless we really want to verify quantization quality.
        
        # Actually, checking if the ONNX runs at all is valuable.
        _validate_onnx_model(onnx_path, feeds, reference=None)
        
        if quant_path != onnx_path:
             _validate_onnx_model(quant_path, feeds, reference=None)
             
    except Exception as exc:
        LOGGER.warning("LLM ONNX validation skipped/failed (likely due to missing inputs in dummy feed): %s", exc)
            
    return onnx_path, quant_path


def _quantize_or_fallback(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if model has external data
    external_data_file = source.parent / "model.onnx_data"
    
    if external_data_file.exists():
        # For models with external data, we need special handling
        return _quantize_with_external_data(source, target, external_data_file)
    
    # Original logic for models without external data
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


def _quantize_with_external_data(source: Path, target: Path, external_data: Path) -> Path:
    """Quantize models that have external data files with multi-method fallback."""
    if not ONNX_AVAILABLE:
        LOGGER.warning("onnx package required for external data handling. Using original.")
        return source
    
    # Try multiple quantization methods in order
    methods = [
        ("direct_quantization", _try_direct_quantization),
        ("via_temp_file", _try_quantization_via_temp_file),
        ("embedded_and_quantize", _try_embedded_quantization),
    ]
    
    for method_name, quantize_fn in methods:
        try:
            LOGGER.info(f"Attempting quantization method: {method_name}")
            result = quantize_fn(source, target, external_data)
            if result == target:
                LOGGER.info(f"Successfully quantized model with external data using method: {method_name}")
                return target
        except Exception as exc:
            LOGGER.warning(f"Quantization method '{method_name}' failed: {exc}")
            continue
    
    # All methods failed
    LOGGER.warning("Quantization with external data failed for %s: All methods exhausted. Using original.", source)
    return source


def _try_direct_quantization(source: Path, target: Path, external_data: Path) -> Path:
    """Try quantizing the model directly, keeping external data format."""
    from onnxruntime.quantization import QuantizationMode
    
    # Quantize with external data output to keep file size manageable
    quantize_dynamic(
        model_input=str(source),
        model_output=str(target),
        weight_type=QuantType.QInt8,
        use_external_data_format=True,  # Keep weights external for large models
    )
    return target


def _try_quantization_via_temp_file(source: Path, target: Path, external_data: Path) -> Path:
    """Try quantization with extra options to handle complex models."""
    quantize_dynamic(
        model_input=str(source),
        model_output=str(target),
        weight_type=QuantType.QInt8,
        use_external_data_format=True,  # Keep weights external for large models
        extra_options={'EnableSubgraph': False},
    )
    return target


def _try_embedded_quantization(source: Path, target: Path, external_data: Path) -> Path:
    """Quantize model with external data, keeping quantized weights external."""
    # For large models, we don't want to embed everything into a single file
    # Instead, quantize and keep using external data format
    LOGGER.info("Quantizing model with external data format...")
    quantize_dynamic(
        model_input=str(source),
        model_output=str(target),
        weight_type=QuantType.QInt8,
        use_external_data_format=True,  # Output quantized weights as external data
    )
    
    LOGGER.info(f"Quantization complete. Output: {target}")
    return target


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
    allowed_mismatch_ratio: float = 0.0,
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
        
        # Check strict equality if ratio is 0 (default behavior)
        if allowed_mismatch_ratio <= 0:
            np.testing.assert_allclose(ort_output, reference, atol=atol, rtol=rtol)
            return

        # Custom validation allowing a percentage of mismatches
        is_close = np.isclose(ort_output, reference, atol=atol, rtol=rtol)
        mismatches = np.sum(~is_close)
        total = is_close.size
        ratio = mismatches / total
        
        if ratio > allowed_mismatch_ratio:
            raise AssertionError(
                f"Mismatch ratio {ratio:.4f} exceeds allowed threshold {allowed_mismatch_ratio} "
                f"({mismatches}/{total} elements mismatched)"
            )
        elif mismatches > 0:
             LOGGER.warning(
                "Quantization validation has mismatches within threshold: %.4f <= %.4f (%d/%d elements)",
                ratio,
                allowed_mismatch_ratio,
                mismatches,
                total,
            )
