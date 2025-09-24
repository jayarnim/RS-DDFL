from tqdm import tqdm
import pandas as pd
import torch
from ..utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_LABEL_COL,
    DEFAULT_PREDICTION_COL,
)


class EarlyStoppingPredictor:
    def __init__(
        self, 
        model, 
        col_user: str=DEFAULT_USER_COL,
        col_item: str=DEFAULT_ITEM_COL,
        col_label: str=DEFAULT_LABEL_COL,
        col_prediction: str=DEFAULT_PREDICTION_COL,
    ):
        # device setting
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(DEVICE)

        self.model = model.to(self.device)
        self.col_user = col_user
        self.col_item = col_item
        self.col_label= col_label
        self.col_prediction = col_prediction

    def predict(
        self,
        dataloader: torch.utils.data.dataloader.DataLoader,
    ):
        # evaluation
        self.model.eval()

        # to save result
        user_idx_list = []
        item_idx_list = []
        label_list = []
        pred_list = []

        iter_obj = tqdm(
            iterable=dataloader, 
            desc=f"EVALUATION",
        )

        for user_idx, item_idx, label in iter_obj:
            # to gpu
            kwargs = dict(
                user_idx=user_idx.to(self.device),
                item_idx=item_idx.to(self.device),
            )

            # predict
            preds = self.model.predict(**kwargs)

            # to cpu & save
            user_idx_list.extend(user_idx.cpu().tolist())
            item_idx_list.extend(item_idx.cpu().tolist())
            label_list.extend(label.cpu().tolist())
            pred_list.extend(preds.cpu().tolist())

        # list -> df
        result = pd.DataFrame(
            {
                self.col_user: user_idx_list,
                self.col_item: item_idx_list,
                self.col_label: label_list,
                self.col_prediction: pred_list,
            }
        )

        return result