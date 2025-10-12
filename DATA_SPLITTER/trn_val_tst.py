from typing import Optional
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from .utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    LEARNING_TYPE,
    SEED,
    HIST_SELECTOR_TYPE,
)
from .data_splitter.python_splitters import python_stratified_split
from .dataloader import pointwise, pairwise, listwise
from . import hist_selector


class TRN_VAL_TST:
    def __init__(
        self, 
        n_users: int, 
        n_items: int,
        learning_type: LEARNING_TYPE="pointwise",
        col_user: str=DEFAULT_USER_COL, 
        col_item: str=DEFAULT_ITEM_COL,
    ):
        """
        Dataset splitter and loader builder for training, validation, testing, and
        leave-one-out (LOO) evaluation in latent factor model experiments
        ------------
        created by @jayarnim

        args:
            n_users (int):
                Total number of users in the dataset.
            n_items (int):
                Total number of items in the dataset.
            learning_type (str):
                Learning paradigm used for training data construction.
                Determines the type of `CustomizedDataLoader` to be instantiated.
                `pointwise`, `pairwise`, `listwise` optional.
        """
        # global attr
        self.n_users = n_users
        self.n_items = n_items
        self.col_user = col_user
        self.col_item = col_item
        self.learning_type = learning_type

        # set up components, dataloader, etc.
        self._set_up_components()

    def get(
        self, 
        origin: pd.DataFrame,
        trn_val_tst_ratio: dict=dict(trn=0.8, val=0.1, tst=0.1),
        neg_per_pos_ratio: dict=dict(trn=4, val=4, tst=99, loo=99),
        batch_size: dict=dict(trn=256, val=256, tst=256, loo=1000),
        hist_selector_type: HIST_SELECTOR_TYPE="default",
        max_hist: Optional[int]=None,
        shuffle: bool=True,
        seed: int=SEED,
    ):
        """
        Splits the input DataFrame into train / validation / test / LOO subsets,
        constructs corresponding DataLoaders,
        and generates historical context information for users and items.

        Args:
            origin (pd.DataFrame):
                The full implicit-feedback dataset containing user-item interactions.  
                Must include columns corresponding to `col_user`, `col_item`, and (optionally) `col_rating` if available.
            trn_val_tst_ratio (dict): `{"trn": float, "val": float, "tst": float}`
                Dictionary specifying the data split ratios for train, validation, and test sets.  
            neg_per_pos_ratio (dict): `{"trn": int, "val": int, "tst": int, "loo": int}`
                Dictionary specifying the number of negative samples per positive instance for each split.  
            batch_size (dict): `{trn: int, val: int, tst: int, loo: int}`
                Batch sizes for each split.  
            hist_selector_type (str):
                Strategy for selecting user/item interaction histories.  
                - `default`: use all past interactions.  
                - `tfidf`: select a subset of informative interactions via TF-IDF weighting.
            max_hist (int):
                Maximum number of historical interactions to retain per user/item.  
                If `None`, all available history is used.

        Returns:
            loaders (dict): `{"trn": dataloader, "val": dataloader, "tst": dataloader, "loo": dataloader}`
                A dictionary containing the constructed DataLoaders.
            user_item_matrix (torch.Tensor): 
                User-item interaction matrix derived from the full dataset.
                (shape: [U+1,I+1])
            hist (dict): `{"user": torch.Tensor, "item": torch.Tensor}`
                Aggregated user and item interaction histories.
                The selection strategy is controlled by `hist_selector_type`.
        """
        kwargs = dict(
            trn_val_tst_ratio=trn_val_tst_ratio,
            neg_per_pos_ratio=neg_per_pos_ratio,
            batch_size=batch_size,
        )
        self._assert_arg_error(**kwargs)

        # split original data
        kwargs = dict(
            origin=origin,
            trn_val_tst_ratio=trn_val_tst_ratio,
            seed=seed,
        )
        splits = self._data_splitter(**kwargs)

        # generate data loaders
        kwargs = dict(
            origin=origin,
            splits=splits, 
            neg_per_pos_ratio=neg_per_pos_ratio, 
            batch_size=batch_size, 
            shuffle=shuffle,
        )
        loaders = self._dataloader_generator(**kwargs)

        # generate user-item interaction matrix
        kwargs = dict(
            data=splits["trn"],
        )
        user_item_matrix = self._user_item_matrix_generator(**kwargs)

        # generate histories
        hist = {}

        kwargs = dict(
            interactions=user_item_matrix,
            hist_selector_type=hist_selector_type,
            max_hist=max_hist,
        )
        hist["user"] = self._histories_generator(**kwargs)

        kwargs = dict(
            interactions=user_item_matrix.T,
            hist_selector_type=hist_selector_type,
            max_hist=max_hist,
        )
        hist["item"] = self._histories_generator(**kwargs)

        return loaders, user_item_matrix, hist

    def _dataloader_generator(self, origin, splits, neg_per_pos_ratio, batch_size, shuffle):
        loaders = {}

        for split_type in ["trn", "val", "tst", "loo"]:
            kwargs = dict(
                origin=origin,
                split=splits[split_type], 
                neg_per_pos_ratio=neg_per_pos_ratio[split_type], 
                batch_size=batch_size[split_type], 
                shuffle=shuffle,
            )
            
            if split_type=="trn":
                loader = self.dataloader_lrn(**kwargs)
            elif split_type=="val":
                loader = self.dataloader_lrn(**kwargs)
            elif split_type=="tst":
                loader = self.dataloader_eval(**kwargs)
            elif split_type=="loo":
                loader = self.dataloader_eval(**kwargs)

            loaders[split_type] = loader
        
        return loaders

    def _user_item_matrix_generator(self, data):
        kwargs = dict(
            size=(self.n_users + 1, self.n_items + 1),
            dtype=torch.int32,
        )
        user_item_matrix = torch.zeros(**kwargs)

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

        user_item_matrix[user_indices, item_indices] = 1

        return user_item_matrix

    def _histories_generator(self, interactions, hist_selector_type, max_hist):
        # drop padding idx
        interactions_unpadded = interactions[:-1, :-1]

        # padding idx
        n_target, n_counterpart = interactions_unpadded.shape

        # select hist per target
        kwargs = dict(
            interactions=interactions_unpadded,
            hist_selector_type=hist_selector_type,
            max_hist=max_hist,
        )
        hist_indices = self._hist_selector(**kwargs)

        # padding
        kwargs = dict(
            sequences=hist_indices, 
            batch_first=True, 
            padding_value=n_counterpart,
        )
        hist_indices_padded = pad_sequence(**kwargs)

        return hist_indices_padded

    def _hist_selector(self, interactions, hist_selector_type, max_hist):
        kwargs = dict(
            interactions=interactions,
            max_hist=max_hist,
        )

        if hist_selector_type=="default":
            return hist_selector.default.selector(interactions)
        elif hist_selector_type=="tfidf":
            return hist_selector.tfidf.selector(**kwargs)
        else:
            raise ValueError(f"Invalid hist_selector_type: {hist_selector_type}")

    def _data_splitter(self, origin, trn_val_tst_ratio, seed):
        split_type = list(trn_val_tst_ratio.keys())
        split_ratio = list(trn_val_tst_ratio.values())

        # for leave one out data set
        loo = (
            origin
            .groupby(self.col_user)
            .sample(n=1, random_state=seed)
            .sort_values(by=self.col_user)
            .reset_index(drop=True)
        )

        # for trn, val, tst data set
        trn_val_tst = (
            origin[~origin[[self.col_user, self.col_item]]
            .apply(tuple, axis=1)
            .isin(set(loo[[self.col_user, self.col_item]]
            .apply(tuple, axis=1)))]
            .reset_index(drop=True)
        )

        # trn_val_tst -> [trn, val, tst]
        kwargs = dict(
            data=trn_val_tst,
            ratio=split_ratio,
            col_user=self.col_user,
            col_item=self.col_item,
            seed=seed,
        )
        split_list = python_stratified_split(**kwargs)

        split_dict = dict(zip(split_type, split_list))
        split_dict["loo"] = loo

        return split_dict

    def _assert_arg_error(self, trn_val_tst_ratio, neg_per_pos_ratio, batch_size):
        CONDITION = (list(trn_val_tst_ratio.keys()) == ["trn", "val", "tst"])
        ERROR_MESSAGE = f"key of trn_val_tst_ratio must be ['trn', 'val', 'tst'], but: {list(trn_val_tst_ratio.keys())}"
        assert CONDITION, ERROR_MESSAGE

        CONDITION = (list(neg_per_pos_ratio.keys()) == ["trn", "val", "tst", "loo"])
        ERROR_MESSAGE = f"key of neg_per_pos_ratio must be ['trn', 'val', 'tst', 'loo'], but: {list(neg_per_pos_ratio.keys())}"
        assert CONDITION, ERROR_MESSAGE

        CONDITION = (list(batch_size.keys()) == ["trn", "val", "tst", "loo"])
        ERROR_MESSAGE = f"key of batch_size must be ['trn', 'val', 'tst', 'loo'], but: {list(batch_size.keys())}"
        assert CONDITION, ERROR_MESSAGE

    def _set_up_components(self):
        self._init_dataloader_lrn()
        self._init_dataloader_eval()

    def _init_dataloader_lrn(self):
        kwargs = dict(
            col_user=self.col_user,
            col_item=self.col_item,
        )
        if self.learning_type=="pointwise":
            self.dataloader_lrn = pointwise.CustomizedDataLoader(**kwargs)
        elif self.learning_type=="pairwise":
            self.dataloader_lrn = pairwise.CustomizedDataLoader(**kwargs)
        elif self.learning_type=="listwise":
            self.dataloader_lrn = listwise.CustomizedDataLoader(**kwargs)
        else:
            raise TypeError(f"Invalid learning_type: {self.learning_type}")

    def _init_dataloader_eval(self):
        kwargs = dict(
            col_user=self.col_user,
            col_item=self.col_item,
        )
        self.dataloader_eval = pointwise.CustomizedDataLoader(**kwargs)