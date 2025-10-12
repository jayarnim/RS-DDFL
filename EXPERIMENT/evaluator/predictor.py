from tqdm import tqdm
from IPython.display import clear_output
from time import perf_counter
from statistics import mean
import pandas as pd
import torch
import torch.nn as nn
from ..utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_PREDICTION_COL,
)
from DATA_SPLITTER.dataloader.pointwise import CustomizedDataLoader


class PerformancePredictor:
    def __init__(
        self, 
        model: nn.Module, 
        col_user: str=DEFAULT_USER_COL,
        col_item: str=DEFAULT_ITEM_COL,
        col_rating: str=DEFAULT_RATING_COL,
        col_prediction: str=DEFAULT_PREDICTION_COL,
    ):
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(DEVICE)

        self.model = model.to(self.device)
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating= col_rating
        self.col_prediction = col_prediction

    @torch.no_grad()
    def __call__(
        self,
        tst_loader: CustomizedDataLoader,
    ):
        # evaluation
        self.model.eval()

        # to save result
        user_idx_list = []
        item_idx_list = []
        label_list = []
        pred_list = []
        computing_cost_list = []

        iter_obj = tqdm(
            iterable=tst_loader, 
            desc=f"TST",
        )

        for user_idx, item_idx, label in iter_obj:
            # to gpu
            kwargs = dict(
                user_idx=user_idx.to(self.device),
                item_idx=item_idx.to(self.device),
            )

            # set starting time for computing cost
            t0 = perf_counter()
            
            # predict
            pred = self.model.predict(**kwargs)
            
            # calculate computing cost
            computing_cost = perf_counter() - t0

            # save
            user_idx_list.extend(user_idx.cpu().tolist())
            item_idx_list.extend(item_idx.cpu().tolist())
            label_list.extend(label.cpu().tolist())
            pred_list.extend(pred.cpu().tolist())
            computing_cost_list.append(computing_cost)

        # list -> df
        result = pd.DataFrame(
            {
                self.col_user: user_idx_list,
                self.col_item: item_idx_list,
                self.col_rating: label_list,
                self.col_prediction: pred_list,
            }
        )

        clear_output(wait=False)

        print(
            "COMPUTING COST FOR INFERENCE",
            f"\t(s/epoch): {sum(computing_cost_list):.4f}",
            f"\t(epoch/s): {1.0/sum(computing_cost_list):.4f}",
            f"\t(s/batch): {mean(computing_cost_list):.4f}",
            f"\t(batch/s): {1.0/mean(computing_cost_list):.4f}",
            sep="\n",
        )

        return result