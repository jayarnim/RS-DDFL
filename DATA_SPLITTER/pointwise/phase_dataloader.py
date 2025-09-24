from itertools import chain
import pandas as pd
from ..utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
)
from .negative_sampling_dataloader import PointwiseNegativeSamplingDataLoader


class DataLoaderCombination:
    def __init__(self, dataloader_list):
        self.dataloader_list = dataloader_list

    def __iter__(self):
        return chain(*self.dataloader_list)

    def __len__(self):
        return sum(len(dataloader) for dataloader in self.dataloader_list)


class PointwisePhaseDataLoader:
    def __init__(
        self,
        origin,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        n_phases=10,
    ):
        self.origin = origin
        self.col_user = col_user
        self.col_item = col_item
        self.n_phases = n_phases

        kwargs = dict(
            origin=self.origin,
            col_user=self.col_user,
            col_item=self.col_item,
        )
        self.dataloader = PointwiseNegativeSamplingDataLoader(**kwargs)

    def get(
        self, 
        data: pd.DataFrame, 
        neg_per_pos: int,
        batch_size: int,
        shuffle: bool=True,
    ):
        # dict: {u_idx: len(hist)}
        user2histlen = (
            self.origin
            .groupby(self.col_user)[self.col_item]
            .count()
            .to_dict()
        )

        # user idx sorted by len(hist)
        sorted_users = sorted(
            user2histlen, 
            key=user2histlen.get,
        )

        # generate phases
        n_total = len(sorted_users)

        phase_user_list = []

        for i in range(self.n_phases):
            start = (i * n_total) // self.n_phases
            end = ((i + 1) * n_total) // self.n_phases
            phase_user = set(sorted_users[start:end])
            phase_user_list.append(phase_user)

        # df per user sorted by hist
        sorted_df_per_phase_list = [
            df for phase in phase_user_list
            for df in [data[data[self.col_user].isin(phase)]]
            if len(phase) > 0 and not df.empty
        ]

        user_dataloader_list = [
            self.dataloader.get(
                data=sorted_df_per_phase, 
                neg_per_pos=neg_per_pos, 
                batch_size=batch_size, 
                shuffle=shuffle,
            )
            for sorted_df_per_phase in sorted_df_per_phase_list
        ]

        return DataLoaderCombination(user_dataloader_list)