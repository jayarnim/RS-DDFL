import torch

def climf(pos, neg):
    diff = neg - pos.unsqueeze(1)
    max_pos_term  = torch.log(torch.sigmoid(pos) + 1e-10)
    min_diff_term = torch.log(1 - torch.sigmoid(diff) + 1e-10).sum(dim=1)
    return -(max_pos_term + min_diff_term)