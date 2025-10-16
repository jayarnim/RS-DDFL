import torch


def bpr(
    pos: torch.Tensor, 
    neg: torch.Tensor,
):
    diff = pos - neg
    return -torch.log(torch.sigmoid(diff)).mean()