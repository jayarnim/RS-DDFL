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
        learning_type: LEARNING_TYPE,
        n_users: int, 
        n_items: int,
        col_user: str=DEFAULT_USER_COL, 
        col_item: str=DEFAULT_ITEM_COL,
    ):
        self.learning_type = learning_type
        self.n_users = n_users
        self.n_items = n_items
        self.col_user = col_user
        self.col_item = col_item

        self._set_up_components()

    def get(
        self, 
        origin: pd.DataFrame,
        trn_val_tst_ratio: dict=dict(trn=0.8, val=0.1, tst=0.1),
        neg_per_pos_ratio: dict=dict(trn=1, val=1, tst=99, loo=99),
        batch_size: dict=dict(trn=256, val=256, tst=256, loo=1000),
        hist_selector_type: HIST_SELECTOR_TYPE="default",
        max_hist: Optional[int]=None,
        shuffle: bool=True,
        seed: int=SEED,
    ):
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
        split_dict = self._data_splitter(**kwargs)

        # generate data loaders
        loaders = []

        for split_type in ["trn", "val", "tst", "loo"]:
            kwargs = dict(
                origin=origin,
                split=split_dict[split_type], 
                neg_per_pos_ratio=neg_per_pos_ratio[split_type], 
                batch_size=batch_size[split_type], 
                shuffle=shuffle,
            )
            
            if split_type=="trn":
                loader = self.dataloader_lrn.get(**kwargs)
            elif split_type=="val":
                loader = self.dataloader_lrn.get(**kwargs)
            elif split_type=="tst":
                loader = self.dataloader_eval.get(**kwargs)
            elif split_type=="loo":
                loader = self.dataloader_eval.get(**kwargs)
            else:
                raise ValueError(f"split type is wrong: {split_type}")
            
            loaders.append(loader)

        # generate user-item interaction matrix
        user_item_matrix = self._user_item_matrix_generator(split_dict["trn"])

        # generate histories
        kwargs = dict(
            interactions=user_item_matrix,
            hist_selector_type=hist_selector_type,
            max_hist=max_hist,
        )
        user_hist = self._histories_generator(**kwargs)

        kwargs = dict(
            interactions=user_item_matrix.T,
            hist_selector_type=hist_selector_type,
            max_hist=max_hist,
        )
        item_hist = self._histories_generator(**kwargs)

        return loaders, user_item_matrix, (user_hist, item_hist)

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

    def _histories_generator(
        self, 
        interactions: torch.Tensor, 
        hist_selector_type: HIST_SELECTOR_TYPE='default',
        max_hist: Optional[int]=None,
    ):
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

    def _hist_selector(
        self,
        interactions: torch.Tensor,
        hist_selector_type: HIST_SELECTOR_TYPE='default',
        max_hist: Optional[int]=None,
    ):
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

    def _data_splitter(
        self,
        origin: pd.DataFrame,
        trn_val_tst_ratio: dict,
        seed: int,
    ):
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
        ERROR_MESSAGE = f"key of neg_per_pos_ratio must be ['trn', 'val', 'tst', 'loo], but: {list(neg_per_pos_ratio.keys())}"
        assert CONDITION, ERROR_MESSAGE

        CONDITION = (list(batch_size.keys()) == ["trn", "val", "tst", "loo"])
        ERROR_MESSAGE = f"key of batch_size must be ['trn', 'val', 'tst', 'loo], but: {list(batch_size.keys())}"
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