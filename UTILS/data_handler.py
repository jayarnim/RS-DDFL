from typing import Optional
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from .constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
)


def rename_columns(
    data: pd.DataFrame,
    col_user: str,
    col_item: str,
    col_rating: Optional[str]=None,
    col_timestamp: Optional[str]=None,
):
    COL_LIST = [col_user, col_item]
    RE_COL_LIST = [DEFAULT_USER_COL, DEFAULT_ITEM_COL]

    if col_rating is not None:
        COL_LIST.append(col_rating)
        RE_COL_LIST.append(DEFAULT_RATING_COL)

    if col_timestamp is not None:
        COL_LIST.append(col_timestamp)
        RE_COL_LIST.append(DEFAULT_TIMESTAMP_COL)

    RENAMES = dict(zip(COL_LIST, RE_COL_LIST))
    
    data = data[COL_LIST]
    data = data.rename(columns=RENAMES)

    return data


def description(
    data: pd.DataFrame, 
    percentaile: float=0.9,
    col_user: str=DEFAULT_USER_COL, 
    col_item: str=DEFAULT_ITEM_COL,
):
    user_counts = data[col_user].value_counts()
    item_counts = data[col_item].value_counts()

    N_USERS = data[col_user].nunique()
    N_ITEMS = data[col_item].nunique()
    TOTAL_INTERACTION = len(data)
    DENSITY = data.shape[0] / (N_USERS * N_ITEMS)
    MAX_USER_INTERACTION = user_counts.max()
    MAX_ITEM_INTERACTION = item_counts.max()
    TOP_PERCENTAILE_USER_INTERACTION = user_counts.quantile(percentaile)
    TOP_PERCENTAILE_ITEM_INTERACTION = item_counts.quantile(percentaile)

    print(
        f"number of user: {N_USERS}",
        f"number of item: {N_ITEMS}",
        f"total interaction: {TOTAL_INTERACTION}",
        f"interaction density: {DENSITY * 100:.4f} %",
        f"max interaction of user: {MAX_USER_INTERACTION}",
        f"max interaction of item: {MAX_ITEM_INTERACTION}",
        f"top {(1-percentaile) * 100:.1f} % interaction of user: {TOP_PERCENTAILE_USER_INTERACTION:.1f}",
        f"top {(1-percentaile) * 100:.1f} % interaction of item: {TOP_PERCENTAILE_ITEM_INTERACTION:.1f}",
        f"mean interaction of user: {TOTAL_INTERACTION // N_USERS}",
        f"mean interaction of item: {TOTAL_INTERACTION // N_ITEMS}",
        sep="\n",
    )


def valid_users(
    data: pd.DataFrame, 
    col_user: str=DEFAULT_USER_COL, 
    min_interaction: int=5,
):
    user_counts = data[col_user].value_counts()
    valid_users = user_counts[user_counts >= min_interaction].index
    return valid_users


def valid_items(
    data: pd.DataFrame, 
    col_item: str=DEFAULT_ITEM_COL, 
    min_interaction: int=5,
):
    item_counts = data[col_item].value_counts()
    valid_items = item_counts[item_counts >= min_interaction].index
    return valid_items


def label_encoding(
    data: pd.DataFrame, 
    col_user: str=DEFAULT_USER_COL, 
    col_item: str=DEFAULT_ITEM_COL,
):
    user_encoder = LabelEncoder()
    data[col_user] = user_encoder.fit_transform(data[col_user])
    user_label = dict(zip(user_encoder.classes_, user_encoder.transform(user_encoder.classes_)))
    
    item_encoder = LabelEncoder()
    data[col_item] = item_encoder.fit_transform(data[col_item])
    item_label = dict(zip(item_encoder.classes_, item_encoder.transform(item_encoder.classes_)))

    return data, user_label, item_label
