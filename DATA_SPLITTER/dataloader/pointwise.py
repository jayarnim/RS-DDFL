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

        self._set_up_components()

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        decision = (idx % (1 + self.neg_per_pos_ratio) == 0)

        if decision==True:
            user, pos = self.user_item_pairs[idx // (1 + self.neg_per_pos_ratio)]
            return user, pos, 1
        else:
            user, _ = self.user_item_pairs[idx // (1 + self.neg_per_pos_ratio)]
            neg = random.choice(self.neg_per_user[user])
            return user, neg, 0

    def _set_up_components(self):
        self._init_entities()
        self._init_candidates()

    def _init_entities(self):
        self.user_list = sorted(self.origin[self.col_user].unique())
        self.item_list = sorted(self.origin[self.col_item].unique())

        self.n_users = len(self.user_list)
        self.n_items = len(self.item_list)

        zip_obj = zip(self.split[self.col_user], self.split[self.col_item])
        self.user_item_pairs = list(zip_obj)
        self.total_samples = len(self.user_item_pairs) * (1 + self.neg_per_pos_ratio)

    def _init_candidates(self):
        self.pos_per_user = {
            user: set(self.origin.loc[self.origin[self.col_user]==user, self.col_item].tolist())
            for user in self.user_list
        }

        self.neg_per_user = {
            user: list(set(self.item_list) - self.pos_per_user[user])
            for user in self.user_list
        }


class CustomizedDataLoader:
    def __init__(
        self,
        col_user: str=DEFAULT_USER_COL,
        col_item: str=DEFAULT_ITEM_COL,
    ):
        self.col_user = col_user
        self.col_item = col_item

    def __call__(
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
        user_list, item_list, label_list = zip(*batch)
        
        user_batch = torch.tensor(user_list, dtype=torch.long)
        item_batch = torch.tensor(item_list, dtype=torch.long)
        label_batch = torch.tensor(label_list, dtype=torch.float32)
        
        return user_batch, item_batch, label_batch