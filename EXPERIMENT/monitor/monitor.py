import copy
import torch
import torch.nn as nn
import pandas as pd
from ..utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_LABEL_COL,
    DEFAULT_PREDICTION_COL,
    DEFAULT_K,
    METRIC_FN_TYPE,
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
from PIPELINE.dataloader.pointwise import CustomizedDataLoader


class EarlyStoppingMonitor:
    def __init__(
        self,
        model: nn.Module,
        patience: int,
        delta: float,
        metric_fn_type: METRIC_FN_TYPE="ndcg",
        col_user: str=DEFAULT_USER_COL,
        col_item: str=DEFAULT_ITEM_COL,
        col_label: str=DEFAULT_LABEL_COL,
        col_prediction: str=DEFAULT_PREDICTION_COL,
        top_k: int=DEFAULT_K,
    ):
        """
        Early Stopping Monitor for Latent Factor Model based on Metrics, not Loss
        -----
        created by @jayarnim

        Args:
            model (nn.Module):
                latent factor model instance.
            patience (int):
                number of epochs to wait for improvement before stopping training early.
            delta (float):
                minimum change in the monitored metric to qualify as an improvement.
            metric_fn_type (str):
                metric functions currently supported are: `hr`, `precision`, `recall`, `map`, `ndcg`. 
        """
        # device setting
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(DEVICE)

        # global attr
        self.model = model.to(self.device)
        self.metric_fn_type = metric_fn_type
        self.patience = patience
        self.delta = delta
        self.col_user = col_user
        self.col_item = col_item
        self.col_label= col_label
        self.col_prediction = col_prediction
        self.top_k = top_k

        self._set_up_components()

    def __call__(
        self,
        loo_loader: CustomizedDataLoader,
        epoch: int,
        n_epochs: int,
    ):
        kwargs = dict(
            loo_loader=loo_loader,
            epoch=epoch,
            n_epochs=n_epochs,
        )
        result = self.predictor(**kwargs)

        rating_true, rating_pred = self._true_pred_seperator(result)

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
        self.stopper(**kwargs)

        return score

    @property
    def should_stop(self):
        return self.stopper.should_stop

    @should_stop.setter
    def should_stop(self, value):
        self.stopper.should_stop = value

    @property
    def get_counter(self):
        return self.stopper.get_counter
    
    @get_counter.setter
    def set_counter(self, value):
        self.stopper.set_counter = value

    @property
    def get_best_epoch(self):
        return self.stopper.get_best_epoch

    @property
    def get_best_score(self):
        return self.stopper.get_best_score

    @property
    def get_best_model_state(self):
        return self.stopper.get_best_model_state

    def _true_pred_seperator(
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

    def _set_up_components(self):
        self._init_metric_fn()
        self._init_predictor()
        self._init_stopper()

    def _init_metric_fn(self):
        if self.metric_fn_type=="hr":
            self.metric_fn = hit_ratio_at_k
        elif self.metric_fn_type=="precision":
            self.metric_fn = precision_at_k
        elif self.metric_fn_type=="recall":
            self.metric_fn_type = recall_at_k
        elif self.metric_fn_type=="map":
            self.metric_fn = map_at_k
        elif self.metric_fn_type=="ndcg":
            self.metric_fn = ndcg_at_k
        else:
            raise ValueError(f"Invalid metric function: {self.metric_fn_type}")

    def _init_predictor(self):
        kwargs = dict(
            model=self.model,
            col_user=self.col_user,
            col_item=self.col_item,
            col_label=self.col_label,
            col_prediction=self.col_prediction,
        )
        self.predictor = EarlyStoppingPredictor(**kwargs)

    def _init_stopper(self):        
        kwargs = dict(
            patience=self.patience,
            delta=self.delta,
        )
        self.stopper = EarlyStopper(**kwargs)