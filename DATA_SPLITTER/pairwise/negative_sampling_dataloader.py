import random
import itertools
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from ..utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
)


class PairwiseNegativeSamplingDataset(Dataset):
    def __init__(
        self, 
        data: pd.DataFrame, 
        neg_items_per_user: dict,
        neg_per_pos: int,
        col_user: str=DEFAULT_USER_COL,
        col_item: str=DEFAULT_ITEM_COL,
    ):
        self.neg_items_per_user = neg_items_per_user
        self.neg_per_pos = neg_per_pos
        self.col_user = col_user
        self.col_item = col_item

        zip_obj = zip(data[self.col_user], data[self.col_item])
        self.user_item_pairs = list(zip_obj)

    def __len__(self):
        return len(self.user_item_pairs)

    def __getitem__(self, idx):
        user, pos = self.user_item_pairs[idx]

        # negative sampling
        kwargs = dict(
            population=self.neg_items_per_user[user],
            k=self.neg_per_pos,     
        )
        neg = random.sample(**kwargs)[0]

        return user, pos, neg


class PairwiseNegativeSamplingDataLoader:
    def __init__(
        self,
        origin: pd.DataFrame,
        col_user: str=DEFAULT_USER_COL,
        col_item: str=DEFAULT_ITEM_COL,
    ):
        self.col_user = col_user
        self.col_item = col_item

        kwargs = dict(
            origin=origin, 
            col_user=self.col_user, 
            col_item=self.col_item,
        )
        self.neg_items_per_user = self._generate_negative_sample_pool(**kwargs)

    def get(
        self, 
        data: pd.DataFrame,
        neg_per_pos: int,
        batch_size: int,
        shuffle: bool=True,
    ):
        CONDITION = neg_per_pos == 1
        ERROR_MESSAGE = "in pairwise data set, neg per pos must be 1:1"
        assert CONDITION, ERROR_MESSAGE

        kwargs = dict(
            data=data, 
            neg_items_per_user=self.neg_items_per_user,
            neg_per_pos=neg_per_pos,
            col_user=self.col_user, 
            col_item=self.col_item,     
        )
        dataset = PairwiseNegativeSamplingDataset(**kwargs)

        kwargs = dict(
            dataset=dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            collate_fn=self._collate,            
        )
        loader = DataLoader(**kwargs)

        return loader

    def _generate_negative_sample_pool(
        self,
        origin: pd.DataFrame, 
        col_user: str=DEFAULT_USER_COL,
        col_item: str=DEFAULT_ITEM_COL,
    ):
        all_users = sorted(origin[col_user].unique())
        all_items = sorted(origin[col_item].unique())
        
        pos_per_user = {
            user: set(origin[origin[col_user] == user][col_item])
            for user in all_users
        }

        neg_items_per_user = {
            user: list(set(all_items) - pos_per_user[user])
            for user in all_users
        }

        return neg_items_per_user

    def _collate(self, batch):
        user_list, pos_list, neg_list = zip(*batch)

        user_batch = torch.tensor(user_list, dtype=torch.long)
        pos_batch = torch.tensor(pos_list, dtype=torch.long)
        neg_batch = torch.tensor(neg_list, dtype=torch.long)

        return user_batch, pos_batch, neg_batch