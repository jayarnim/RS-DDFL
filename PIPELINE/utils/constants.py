from UTILS.constants import(
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_LABEL_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_PREDICTION_COL,
    COL_DICT,
    DEFAULT_K,
    DEFAULT_THRESHOLD,
    SEED,
)

# Customized
from typing import Literal

LEARNING_TYPE = Literal["pointwise", "pairwise", "listwise"]
HIST_SELECTOR_TYPE = Literal["default", "tfidf"]