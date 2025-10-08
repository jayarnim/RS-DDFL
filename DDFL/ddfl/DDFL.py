import torch
import torch.nn as nn
from . import MaFL, MeFL


class Module(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int,
        alpha: int,
        hidden: list,
        dropout: float,
        interactions: torch.Tensor, 
    ):
        super(Module, self).__init__()

        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # global attr
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.alpha = alpha
        self.hidden = hidden
        self.dropout = dropout
        self.register_buffer(
            name="interactions", 
            tensor=interactions,
        )

        # generate layers
        self._init_layers()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        user_idx: (B,)
        item_idx: (B,)
        """
        return self.score(user_idx, item_idx)

    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        user_idx: (B,)
        item_idx: (B,)
        """
        with torch.no_grad():
            logit = self.score(user_idx, item_idx)
            pred = torch.sigmoid(logit)
        return pred

    def score(self, user_idx, item_idx):
        pred_vector_matching = self.MaFL.ml(user_idx, item_idx)
        pred_vector_metric = self.MeFL.ml(user_idx, item_idx)

        kwargs = dict(
            tensors=(pred_vector_matching, pred_vector_metric), 
            dim=-1,
        )
        pred_vector = torch.cat(**kwargs)
        logit = self.logit_layer(pred_vector).squeeze(-1)

        return logit

    def _set_up_components(self):
        self._create_modules()
        self._create_layers()

    def _create_modules(self):
        kwargs = dict(
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.n_factors,
            hidden=self.hidden,
            dropout=self.dropout,
            interactions=self.interactions,
        )
        self.MaFL = MaFL.Module(**kwargs)

        kwargs = dict(
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.n_factors*2,
            alpha=self.alpha,
            hidden=self.hidden,
            dropout=self.dropout,
            interactions=self.interactions,
        )
        self.MeFL = MeFL.Module(**kwargs)

    def _create_layers(self):
        kwargs = dict(
            in_features=self.hidden[-1]*2,
            out_features=1,
        )
        self.logit_layer = nn.Linear(**kwargs)