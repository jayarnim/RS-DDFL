from typing import Optional, Literal
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfTransformer
from torch.nn.utils.rnn import pad_sequence
from ..utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    LOADING_TYPE,
)
from ..utils.python_splitters import python_stratified_split
from .negative_sampling_dataloader import PointwiseNegativeSamplingDataLoader
from .curriculum_dataloader import PointwiseCurriculumDataLoader
from .userpair_dataloader import PointwiseUserpairDataLoader
from .phase_dataloader import PointwisePhaseDataLoader


class DataSplitter:
    def __init__(
        self, 
        origin: pd.DataFrame,
        n_users: int, 
        n_items: int,
        n_phases: Optional[int]=None,
        col_user: str=DEFAULT_USER_COL, 
        col_item: str=DEFAULT_ITEM_COL,
        loading_type: LOADING_TYPE="general",
    ):
        self.origin = origin
        self.n_users = n_users
        self.n_items = n_items
        self.n_phases = n_phases
        self.col_user = col_user
        self.col_item = col_item

        kwargs = dict(
            origin=self.origin,
            col_user=self.col_user,
            col_item=self.col_item,
        )
        if loading_type=="general":
            self.dataloader = PointwiseNegativeSamplingDataLoader(**kwargs)
        elif loading_type=="curriculum":
            self.dataloader = PointwiseCurriculumDataLoader(**kwargs)
        elif loading_type=="userpair":
            self.dataloader = PointwiseUserpairDataLoader(**kwargs)
        elif loading_type=="phase":
            self.dataloader = PointwisePhaseDataLoader(**kwargs, n_phases=self.n_phases)
        else:
            raise TypeError(f"Invalid loading_type: {loading_type}")

    def get(
        self, 
        trn_val_tst_ratio: list=[0.8, 0.1, 0.1],
        neg_per_pos: list=[4, 4, 99, 99],
        batch_size: list=[256, 256, 256, 1000],
        max_hist: Optional[int]=None,
        shuffle: bool=True,
        seed: int=42,
    ):
        # split original data
        kwargs = dict(
            trn_val_tst_ratio=trn_val_tst_ratio,
            seed=seed,
        )
        trn, val, tst, loo = self._data_splitter(**kwargs)
        split_list = [trn, val, tst, loo]

        # generate data loaders
        loaders = []
        zip_obj = zip(split_list, neg_per_pos, batch_size)

        for split, split_neg_per_pos, split_batch in zip_obj:
            kwargs = dict(
                data=split, 
                neg_per_pos=split_neg_per_pos, 
                batch_size=split_batch, 
                shuffle=shuffle,
            )
            loader = self.dataloader.get(**kwargs)
            loaders.append(loader)

        # generate user-item interaction matrix
        interactions = self._interactions_generator(trn)

        # generate histories
        kwargs = dict(
            interactions=interactions,
            target="user",
            max_hist=max_hist,
        )
        user_hist = self._hist_generator(**kwargs)

        kwargs = dict(
            interactions=interactions,
            target="item",
            max_hist=max_hist,
        )
        item_hist = self._hist_generator(**kwargs)

        return loaders, interactions, (user_hist, item_hist)

    def _interactions_generator(self, data):
        kwargs = dict(
            size=(self.n_users + 1, self.n_items + 1),
            dtype=torch.int32,
        )
        interactions = torch.zeros(**kwargs)

        kwargs = dict(
            data=data[self.col_user].values, 
            dtype=torch.long,
        )
        user_indices = torch.tensor(**kwargs)
        kwargs = dict(
            data=data[self.col_item].values, 
            dtype=torch.long,
        )
        item_indices = torch.tensor(**kwargs)

        interactions[user_indices, item_indices] = 1

        return interactions

    def _hist_generator(
        self, 
        interactions: torch.Tensor, 
        target: Literal['user', 'item'],
        max_hist: Optional[int]=None,
    ):
        interactions = interactions if target=="user" else interactions.T
        num_target = self.n_users if target=="user" else self.n_items
        num_counterpart = self.n_items if target=="user" else self.n_users

        tfidf_dict = self._tfidf(interactions) if max_hist is not None else None
        pos_per_target_list = []

        for target_idx in range(num_target):
            # user row (interaction 벡터)
            target_row = interactions[target_idx]

            # interacted counterpart indices
            counterparts = torch.nonzero(target_row, as_tuple=False).squeeze(-1)

            # interaction X -> padding idx
            if counterparts.numel() == 0:
                counterparts = torch.tensor([num_counterpart], dtype=torch.long)
            # interaction O
            else:
                # top-k based on tf-idf score
                if max_hist is not None and len(counterparts) > max_hist:
                    # scores
                    kwargs = dict(
                        data=[tfidf_dict.get((int(target_idx), int(counterpart_idx)), 0.0) for counterpart_idx in counterparts],
                        dtype=torch.float32,
                    )
                    scores = torch.tensor(**kwargs)
                    # top-k idx
                    top_k_vals, top_k_indices = torch.topk(scores, k=max_hist)
                    # top-k idx selection
                    counterparts = counterparts[top_k_indices]

            pos_per_target_list.append(counterparts)

        # padding
        kwargs = dict(
            sequences=pos_per_target_list, 
            batch_first=True, 
            padding_value=num_counterpart,
        )
        pos_per_target_padded = pad_sequence(**kwargs)

        return pos_per_target_padded

    def _tfidf(
        self, 
        interactions: torch.Tensor,
    ):
        # drop padding idx
        interactions_unpadded = interactions[:-1, :-1]

        # compute tfidf
        tfidf = TfidfTransformer(norm=None)
        tfidf_matrix = tfidf.fit_transform(interactions_unpadded)

        # sparse matrix -> dict: {(u_idx, i_idx): tf-idf score}
        tfidf_dict = {}
        rows, cols = tfidf_matrix.nonzero()
        for row, col in zip(rows, cols):
            tfidf_dict[(row, col)] = tfidf_matrix[row, col]

        return tfidf_dict

    def _data_splitter(
        self,
        trn_val_tst_ratio: list,
        seed: int,
    ):
        # for leave one out data set
        loo = (
            self.origin
            .groupby(self.col_user)
            .sample(n=1, random_state=seed)
            .sort_values(by=self.col_user)
            .reset_index(drop=True)
        )

        # for trn, val, tst data set
        trn_val_tst = (
            self.origin[~self.origin[[self.col_user, self.col_item]]
            .apply(tuple, axis=1)
            .isin(set(loo[[self.col_user, self.col_item]]
            .apply(tuple, axis=1)))]
            .reset_index(drop=True)
        )

        # trn_val_tst -> [trn, val, tst]
        kwargs = dict(
            data=trn_val_tst,
            ratio=trn_val_tst_ratio,
            col_user=self.col_user,
            col_item=self.col_item,
            seed=seed,
        )
        trn, val, tst = python_stratified_split(**kwargs)

        return trn, val, tst, loo