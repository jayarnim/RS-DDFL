import torch.nn.functional as F

def bce(logit, label):
    return F.binary_cross_entropy_with_logits(logit, label)