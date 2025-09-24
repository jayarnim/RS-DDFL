import torch

def bpr_func(pos, neg):
    diff = pos - neg
    return -torch.log(torch.sigmoid(diff)).mean()