import pandas as pd
from ..utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_PREDICTION_COL,
    DEFAULT_K,
)
from ..msr.python_evaluation import (
    hit_ratio_at_k,
    precision_at_k, 
    recall_at_k,
    map_at_k, 
    ndcg_at_k, 
)

class PerformanceEvaluator:
    def __init__(
        self, 
        result: pd.DataFrame,
        col_user: str=DEFAULT_USER_COL,
        col_item: str=DEFAULT_ITEM_COL,
        col_rating: str=DEFAULT_RATING_COL,
        col_prediction: str=DEFAULT_PREDICTION_COL,
    ):
        self.result = result
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating
        self.col_prediction = col_prediction

        self._sep_true_pred()

    def evaluate(
        self,
        top_k_list: list=[5, 10, 15, 20, 25, 50, 100],
    ):
        eval_list = []

        for TOP_K in top_k_list:
            eval = self.top_k(TOP_K)
            eval_list.append(eval)

        return pd.DataFrame(eval_list)

    def top_k(self, k):
        kwargs = dict(
            rating_true=self.rating_true,
            rating_pred=self.rating_pred.head(k),
            col_user=self.col_user,
            col_item=self.col_item,
            col_rating=self.col_rating,
            col_prediction=self.col_prediction,
            k=k,
        )

        hr_k = hit_ratio_at_k(**kwargs)
        prec_k = precision_at_k(**kwargs)
        rec_k = recall_at_k(**kwargs)
        map_k = map_at_k(**kwargs)
        ndcg_k = ndcg_at_k(**kwargs)

        return dict(
            top_k=k,
            hit_ratio=hr_k, 
            precision=prec_k, 
            recall=rec_k, 
            map=map_k, 
            ndcg=ndcg_k,
        )

    def _sep_true_pred(self):
        TRUE_COL_LIST = [self.col_user, self.col_item, self.col_rating]
        PRED_COL_LIST = [self.col_user, self.col_item, self.col_prediction]

        self.rating_true = (
            self.result[TRUE_COL_LIST]
            [self.result[self.col_rating]==1]
            .sort_values(by=self.col_user, ascending=True)
        )

        self.rating_pred = (
            self.result[PRED_COL_LIST]
            .sort_values(by=[self.col_user, self.col_prediction], ascending=[True, False], kind='stable')
            .groupby(self.col_user)
        )