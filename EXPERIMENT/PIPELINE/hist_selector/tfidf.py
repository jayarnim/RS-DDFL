import torch
from sklearn.feature_extraction.text import TfidfTransformer


def selector(
    interactions: torch.Tensor,
    max_hist: int,
):
    # padding idx
    n_target, n_counterpart = interactions.shape

    # compute tfidf
    tfidf = TfidfTransformer(norm=None)
    tfidf_matrix = tfidf.fit_transform(interactions)

    # ndarray -> tensor
    kwargs = dict(
        data=tfidf_matrix.toarray(),
        dtype=torch.float32,
    )
    tfidf_matrix_dense = torch.tensor(**kwargs)

    # select top-k indices
    topk_indices = []
    for row in range(len(tfidf_matrix_dense)):
        hist_count = int(interactions[row].sum().item())
        if hist_count == 0:
            indices = torch.tensor([n_counterpart], dtype=torch.long)
            topk_indices.append(indices)
        elif hist_count <= max_hist:
            indices = interactions[row].nonzero(as_tuple=True)[0]
            topk_indices.append(indices.to(torch.long))
        else:
            vals, indices = torch.topk(tfidf_matrix_dense[row], k=max_hist)
            topk_indices.append(indices.to(torch.long))

    return topk_indices