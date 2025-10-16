# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Default column names
DEFAULT_USER_COL = "userId"
DEFAULT_ITEM_COL = "itemId"
DEFAULT_RATING_COL = "label"
DEFAULT_LABEL_COL = "label"
DEFAULT_TIMESTAMP_COL = "timestamp"
DEFAULT_PREDICTION_COL = "prediction"
COL_DICT = dict(
    col_user=DEFAULT_USER_COL, 
    col_item=DEFAULT_ITEM_COL, 
    col_rating=DEFAULT_RATING_COL, 
    col_prediction=DEFAULT_PREDICTION_COL,
)

# Filtering variables
DEFAULT_K = 10
DEFAULT_THRESHOLD = 10

# Other
SEED = 42
