import torch
import torch.nn as nn
from ..utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_PREDICTION_COL,
    DEFAULT_K,
)
from .predictor import PerformancePredictor
from .metrics import MetricsComputer
from ...PIPELINE.dataloader.pointwise import CustomizedDataLoader


class PerformanceEvaluator:
    def __init__(
        self, 
        model: nn.Module, 
        col_user: str=DEFAULT_USER_COL,
        col_item: str=DEFAULT_ITEM_COL,
        col_rating: str=DEFAULT_RATING_COL,
        col_prediction: str=DEFAULT_PREDICTION_COL,
    ):
        """
        Performance Evaluator for Latent Factor Model
        -----
        created by @jayarnim

        Args:
            model (nn.Module):
                latent factor model instance.
        """
        # device setting
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(DEVICE)

        # global attr
        self.model = model.to(self.device)
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating
        self.col_prediction = col_prediction

        # set up components, predictor, metrics computer, etc.
        self._set_up_components()

    def evaluate(
        self,
        tst_loader: CustomizedDataLoader,
        top_k_list: list=[5, 10, 15, 20, 25, 50, 100],
    ):
        """
        evaluation pipeline launcher method

        Args:
            tst_loader (CustomizedDataLoader):
                DataLoader for the test set.
            top_k_list (list):
                List of cutoff values (K) for which performance metrics are computed.
        """
        result = self.predictor(tst_loader)
        
        rating_true, rating_pred = self._true_pred_seperator(result)

        kwargs = dict(
            rating_true=rating_true,
            rating_pred=rating_pred,
            top_k_list=top_k_list,
        )
        metrics_sheet = self.metrics(**kwargs)

        return metrics_sheet

    def _true_pred_seperator(self, result):
        TRUE_COL_LIST = [self.col_user, self.col_item, self.col_rating]
        PRED_COL_LIST = [self.col_user, self.col_item, self.col_prediction]

        rating_true = (
            result[TRUE_COL_LIST]
            [result[self.col_rating]==1]
            .sort_values(by=self.col_user, ascending=True)
        )

        rating_pred = (
            result[PRED_COL_LIST]
            .sort_values(by=[self.col_user, self.col_prediction], ascending=[True, False], kind='stable')
            .groupby(self.col_user)
        )

        return rating_true, rating_pred

    def _set_up_components(self):
        self._init_predictor()
        self._init_metrics()

    def _init_predictor(self):
        kwargs = dict(
            model=self.model,
            col_user=self.col_user,
            col_item=self.col_item,
            col_rating=self.col_rating,
            col_prediction=self.col_prediction,
        )
        self.predictor = PerformancePredictor(**kwargs)

    def _init_metrics(self):
        kwargs = dict(
            col_user=self.col_user,
            col_item=self.col_item,
            col_rating=self.col_rating,
            col_prediction=self.col_prediction,
        )
        self.metrics = MetricsComputer(**kwargs)