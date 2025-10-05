import torch

def bpr(pos, neg):
    diff = pos - neg
    return -torch.log(torch.sigmoid(diff)).mean()