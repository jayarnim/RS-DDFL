import copy
import torch
import pandas as pd
from ..utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_LABEL_COL,
    DEFAULT_PREDICTION_COL,
    DEFAULT_K,
    METRIC_FN,
)
from ..msr.python_evaluation import (
    hit_ratio_at_k,
    precision_at_k, 
    recall_at_k,
    map_at_k, 
    ndcg_at_k, 
)
from .early_stopper import EarlyStopper
from .predictor import EarlyStoppingPredictor


class EarlyStoppingMonitor:
    def __init__(
        self,
        model,
        metric_fn: METRIC_FN,
        patience: int,
        min_delta: float,
        col_user: str=DEFAULT_USER_COL,
        col_item: str=DEFAULT_ITEM_COL,
        col_label: str=DEFAULT_LABEL_COL,
        col_prediction: str=DEFAULT_PREDICTION_COL,
        top_k: int=DEFAULT_K,
    ):
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(DEVICE)

        self.model = model.to(self.device)
        self.patience = patience
        self.min_delta = min_delta
        self.col_user = col_user
        self.col_item = col_item
        self.col_label= col_label
        self.col_prediction = col_prediction
        self.top_k = top_k

        if metric_fn=="hr":
            self.metric_fn = hit_ratio_at_k
        elif metric_fn=="precision":
            self.metric_fn = precision_at_k
        elif metric_fn=="recall":
            self.metric_fn = recall_at_k
        elif metric_fn=="map":
            self.metric_fn = map_at_k
        elif metric_fn=="ndcg":
            self.metric_fn = ndcg_at_k
        else:
            ValueError(f"Invalid metric function: {metric_fn}")

        kwargs = dict(
            model=self.model,
            col_user=self.col_user,
            col_item=self.col_item,
            col_label=self.col_label,
            col_prediction=self.col_prediction,
        )
        self.predictor = EarlyStoppingPredictor(**kwargs)
        
        kwargs = dict(
            patience=self.patience,
            min_delta=self.min_delta,
        )
        self.stopper = EarlyStopper(**kwargs)

    def monitor(
        self,
        dataloader: torch.utils.data.dataloader.DataLoader,
        epoch: int,
    ):
        result = self.predictor.predict(dataloader)

        rating_true, rating_pred = self._sep_true_pred(result)

        kwargs = dict(
            rating_true=rating_true,
            rating_pred=rating_pred,
            col_user=self.col_user,
            col_item=self.col_item,
            col_rating=self.col_label,
            col_prediction=self.col_prediction,
            k=self.top_k,
        )
        score = self.metric_fn(**kwargs)

        kwargs = dict(
            current_score=score, 
            current_epoch=epoch,
            current_model_state=copy.deepcopy(self.model.state_dict()),
        )
        self.stopper.check(**kwargs)

        return score

    def _sep_true_pred(
        self,
        result: pd.DataFrame,
    ):
        TRUE_COL_LIST = [self.col_user, self.col_item, self.col_label]
        PRED_COL_LIST = [self.col_user, self.col_item, self.col_prediction]

        rating_true = (
            result[TRUE_COL_LIST]
            [result[self.col_label]==1]
            .sort_values(by=self.col_user, ascending=True)
        )

        rating_pred = (
            result[PRED_COL_LIST]
            .sort_values(by=[self.col_user, self.col_prediction], ascending=[True, False], kind='stable')
            .groupby(self.col_user)
            .head(self.top_k)
        )

        return rating_true, rating_pred