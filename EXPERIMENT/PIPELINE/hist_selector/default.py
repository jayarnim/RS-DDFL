import torch


def selector(
    interactions: torch.Tensor,
):
    # drop padding idx
    interactions_unpadded = interactions[:-1, :-1]

    rows, cols = interactions_unpadded.nonzero(as_tuple=True)

    hist_indices = [[] for _ in range(len(interactions_unpadded))]
    for r, c in zip(rows.tolist(), cols.tolist()):
        hist_indices[r].append(c)

    hist_indices = [
        torch.tensor(indices, dtype=torch.long)
        for indices in hist_indices
    ]

    return hist_indices