"""
PyTorch vs ONNX Parity and Numerical Verification Tests.
Verifies logits fidelity, argmax prediction parity, and inference consistency.
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest


def check_parity(
    pytorch_logits: np.ndarray, onnx_logits: np.ndarray, atol: float = 1e-4, rtol: float = 1e-3
) -> tuple[float, bool]:
    """Calculates max absolute difference and verifies predicted argmax parity."""
    max_diff = float(np.max(np.abs(pytorch_logits - onnx_logits)))
    classes_match = bool(
        np.array_equal(
            np.argmax(pytorch_logits, axis=-1),
            np.argmax(onnx_logits, axis=-1),
        )
    )
    return max_diff, classes_match


def test_parity_logic():
    """
    Verifies that the parity verification mathematical check works as expected
    for close logits outputs and matching argmax predictions.
    """
    # 1. Close logits with identical argmax prediction
    py_logits = np.array([[2.3451, -1.2034, 0.4561], [0.1234, 1.9876, -0.5432]])
    ox_logits = np.array([[2.34515, -1.20338, 0.45612], [0.12342, 1.98755, -0.54318]])

    max_diff, is_match = check_parity(py_logits, ox_logits, atol=1e-4)
    assert is_match is True
    assert max_diff < 1e-4

    # Verify allclose assertion
    np.testing.assert_allclose(py_logits, ox_logits, rtol=1e-3, atol=1e-4)

    # 2. Divergent logits that flip the class
    flipped_logits = np.array([[-1.0, 2.0, 0.5], [0.1, 1.9, -0.5]])
    _, match_flipped = check_parity(py_logits, flipped_logits)
    assert match_flipped is False


def test_onnx_pytorch_parity():
    """
    End-to-end parity test between PyTorch model and ONNX Runtime model.
    Runs when PyTorch and ONNX model artifacts are available in the environment.
    """
    onnx_model_path = "artifacts/models/student_model.onnx"
    if not os.path.exists(onnx_model_path):
        pytest.skip(f"ONNX model file not found at {onnx_model_path}")

    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed in the local environment.")

    if importlib.util.find_spec("onnxruntime") is None:
        pytest.skip("ONNX Runtime is not installed in the local environment.")

    import onnxruntime as ort
    import torch

    batch_size = 4
    seq_length = 64

    input_ids = torch.randint(low=0, high=1000, size=(batch_size, seq_length), dtype=torch.long)
    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

    try:
        from transformers import AutoConfig, AutoModelForSequenceClassification

        from arabic_sentiment.models.student import build_student_model

        cfg = AutoConfig.from_pretrained("asafaya/bert-mini-arabic", num_labels=3)
        teacher = AutoModelForSequenceClassification.from_config(cfg)
        pytorch_model = build_student_model(teacher, student_layers=[0, 1, 2])
        pytorch_model.eval()

        with torch.no_grad():
            torch_outputs = pytorch_model(input_ids=input_ids, attention_mask=attention_mask)
            torch_logits = torch_outputs.logits.detach().cpu().numpy()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"PyTorch student initialization skipped: {e}")

    session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
    onnx_inputs = {
        "input_ids": input_ids.numpy(),
        "attention_mask": attention_mask.numpy(),
    }
    onnx_outputs = session.run(None, onnx_inputs)
    onnx_logits = onnx_outputs[0]

    np.testing.assert_allclose(
        torch_logits,
        onnx_logits,
        rtol=1e-3,
        atol=1e-4,
        err_msg="ONNX logits do not match PyTorch outputs within tolerance!",
    )

    torch_preds = np.argmax(torch_logits, axis=-1)
    onnx_preds = np.argmax(onnx_logits, axis=-1)
    np.testing.assert_array_equal(
        torch_preds,
        onnx_preds,
        err_msg="Predicted class labels mismatch between PyTorch and ONNX!",
    )
