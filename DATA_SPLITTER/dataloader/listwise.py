import random
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from ..utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
)


class CustomizedDataset(Dataset):
    def __init__(
        self, 
        origin: pd.DataFrame,
        split: pd.DataFrame,
        neg_per_pos_ratio: int,
        col_user: str=DEFAULT_USER_COL,
        col_item: str=DEFAULT_ITEM_COL,
    ):
        self.origin = origin
        self.split = split
        self.neg_per_pos_ratio = neg_per_pos_ratio
        self.col_user = col_user
        self.col_item = col_item

        self._setup_components()

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        user, pos = self.user_item_pairs[idx]
        kwargs = dict(
            population=self.neg_per_user[user],
            k=self.neg_per_pos_ratio,
        )
        neg_list = random.sample(**kwargs)
        return user, pos, neg_list

    def _setup_components(self):
        self.user_list = sorted(self.origin[self.col_user].unique())
        self.item_list = sorted(self.origin[self.col_item].unique())

        self.n_users = len(self.user_list)
        self.n_items = len(self.item_list)

        self.pos_per_user = {
            user: set(self.origin.loc[self.origin[self.col_user]==user, self.col_item].tolist())
            for user in self.user_list
        }

        self.neg_per_user = {
            user: list(set(self.item_list) - self.pos_per_user[user])
            for user in self.user_list
        }

        zip_obj = zip(self.split[self.col_user], self.split[self.col_item])
        self.user_item_pairs = list(zip_obj)
        self.total_samples = len(self.user_item_pairs)


class CustomizedDataLoader:
    def __init__(
        self,
        col_user: str=DEFAULT_USER_COL,
        col_item: str=DEFAULT_ITEM_COL,
    ):
        self.col_user = col_user
        self.col_item = col_item

    def get(
        self, 
        origin: pd.DataFrame,
        split: pd.DataFrame,
        neg_per_pos_ratio: int,
        batch_size: int,
        shuffle: bool=True,
    ):
        kwargs = dict(
            origin=origin,
            split=split, 
            neg_per_pos_ratio=neg_per_pos_ratio,
            col_user=self.col_user, 
            col_item=self.col_item,     
        )
        dataset = CustomizedDataset(**kwargs)

        kwargs = dict(
            dataset=dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            collate_fn=self._collate,            
        )
        loader = DataLoader(**kwargs)

        return loader

    def _collate(self, batch):
        user_list, pos_list, neg_list = zip(*batch)
        
        user_batch = torch.tensor(user_list, dtype=torch.long)          # (B,)
        pos_batch  = torch.tensor(pos_list, dtype=torch.long)           # (B,)
        neg_batch  = torch.tensor(neg_list, dtype=torch.long)           # (B, N)
        
        return user_batch, pos_batch, neg_batch