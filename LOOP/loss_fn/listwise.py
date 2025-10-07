import torch.nn.functional as F


def climf(pos, neg):
    diff = neg - pos.unsqueeze(1)
    max_pos_term  = F.logsigmoid(pos)
    min_diff_term = F.logsigmoid(-diff).sum(dim=1)
    return -(max_pos_term + min_diff_term).mean()