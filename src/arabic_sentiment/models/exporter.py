import os

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import nn
from transformers import AutoTokenizer


class OnnxWrapper(nn.Module):
    """Wraps model to output only the logits tensor for ONNX export."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits


class ModelExporter:
    """Handles ONNX export (FP32) and Dynamic INT8 Quantization."""

    def __init__(self, tokenizer_name: str, max_length: int = 128, opset_version: int = 17):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.opset_version = opset_version

    def export_onnx(
        self,
        model: nn.Module,
        output_path: str,
        sample_text: str = "المنتج ممتاز والتوصيل سريع",
    ) -> str:
        """Exports PyTorch model to ONNX graph with dynamic axes."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        model = model.to("cpu").eval()
        wrapper = OnnxWrapper(model)

        # Dummy inputs for tracing
        dummy = self.tokenizer(
            [sample_text] * 2,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        dummy_inputs = (dummy["input_ids"], dummy["attention_mask"])

        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        }

        try:
            torch.onnx.export(
                wrapper,
                dummy_inputs,
                output_path,
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes=dynamic_axes,
                opset_version=self.opset_version,
                do_constant_folding=True,
                dynamo=False,
            )
        except TypeError:
            torch.onnx.export(
                wrapper,
                dummy_inputs,
                output_path,
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes=dynamic_axes,
                opset_version=self.opset_version,
                do_constant_folding=True,
            )

        return output_path

    def quantize_int8(self, onnx_fp32_path: str, onnx_int8_path: str) -> str:
        """Quantizes an FP32 ONNX model to dynamic INT8."""
        os.makedirs(os.path.dirname(os.path.abspath(onnx_int8_path)), exist_ok=True)
        quantize_dynamic(
            model_input=onnx_fp32_path,
            model_output=onnx_int8_path,
            weight_type=QuantType.QInt8,
        )
        return onnx_int8_path
