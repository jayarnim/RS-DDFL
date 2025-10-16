import torch
import torch.nn.functional as F


def bce(
    logit: torch.Tensor, 
    label: torch.Tensor,
):
    return F.binary_cross_entropy_with_logits(logit, label)