import pandas as pd
from ..utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
)
from .negative_sampling_dataloader import PointwiseNegativeSamplingDataLoader


class PointwiseCurriculumDataLoader:
    def __init__(
        self,
        origin,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
    ):
        self.origin = origin
        self.col_user = col_user
        self.col_item = col_item

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
        shuffle: bool=False,
    ):
        CONDITION = shuffle==False
        ERROR_MESSAGE = "if use curriculum dataloader, shuffle must be False"
        assert CONDITION, ERROR_MESSAGE

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

        # df per user sorted by hist
        sorted_df_per_user_list = [
            data[data[self.col_user] == user]
            for user in sorted_users
        ]
        kwargs = dict(
            objs=sorted_df_per_user_list, 
            ignore_index=True,
        )
        sorted_df_concat = pd.concat(**kwargs)

        # generate curriculum dataloader
        kwargs = dict(
            data=sorted_df_concat, 
            neg_per_pos=neg_per_pos, 
            batch_size=batch_size,
            shuffle=shuffle,
        )
        loader = self.dataloader.get(**kwargs)

        return loader