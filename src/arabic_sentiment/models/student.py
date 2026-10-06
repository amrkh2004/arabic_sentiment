import copy

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForSequenceClassification


def build_student_model(
    teacher_model: nn.Module,
    student_layers: list[int] | None = None,
) -> nn.Module:
    """
    Initializes a 6-layer student model by copying alternating layers
    from the 12-layer fine-tuned teacher model (DistilBERT style).
    """
    if student_layers is None:
        student_layers = [0, 2, 4, 6, 8, 10]
    config = copy.deepcopy(teacher_model.config)
    config.num_hidden_layers = len(student_layers)

    student = AutoModelForSequenceClassification.from_config(config)

    # Copy embeddings
    student.bert.embeddings.load_state_dict(teacher_model.bert.embeddings.state_dict())

    # Copy chosen encoder layers
    for s_idx, t_idx in enumerate(student_layers):
        student.bert.encoder.layer[s_idx].load_state_dict(
            teacher_model.bert.encoder.layer[t_idx].state_dict()
        )

    # Copy pooler and classifier heads
    if (
        hasattr(student.bert, "pooler")
        and hasattr(teacher_model.bert, "pooler")
        and teacher_model.bert.pooler is not None
    ):
        student.bert.pooler.load_state_dict(teacher_model.bert.pooler.state_dict())

    student.classifier.load_state_dict(teacher_model.classifier.state_dict())
    return student


def compute_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.5,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Computes combined knowledge distillation loss:
    L = alpha * CrossEntropy(student, labels) + (1 - alpha) * T^2 * KL(softmax(s/T) || softmax(t/T))
    """
    ce_loss = F.cross_entropy(student_logits, labels, weight=class_weights)
    kl_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature * temperature)

    return alpha * ce_loss + (1.0 - alpha) * kl_loss
