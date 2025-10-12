import torch
import torch.nn as nn
from . import MaFL, MeFL


class Module(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int,
        hidden: list,
        dropout: float,
        alpha: float,
        interactions: torch.Tensor, 
    ):
        """
        DDFL: a deep dual function learning-based model for recommender systems (Shah et al., 2020)
        -----
        Implements the base structure of Deep Dual Function Learning-based Model (DDFL),
        MLP & history, distance embedding based latent factor model,
        combining a Metric Function Learning (MeFL) and a Matching Function Learning (MaFL)
        to learn user-item distance & user-item interactions.

        Args:
            n_users (int): 
                total number of users in the dataset, U.
            n_items (int): 
                total number of items in the dataset, I.
            n_factors (int): 
                dimensionality of user and item latent representation vectors, K.
            hidden (list): 
                layer dimensions for the distance & matching function. 
                (e.g., [64, 32, 16, 8])
            dropout (float): 
                dropout rate applied to MLP layers for regularization.
            alpha (float): 
                distance factor.
            interaction (torch.Tensor): 
                user-item interaction matrix, masked evaluation datasets. 
                (shape: [U+1, I+1])
        """
        super().__init__()

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
        self._set_up_components()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Training Method

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])
        
        Returns:
            logit (torch.Tensor): (u,i) pair interaction logit (shape: [B,])
        """
        return self.score(user_idx, item_idx)

    @torch.no_grad()
    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Evaluation Method

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])

        Returns:
            prob (torch.Tensor): (u,i) pair interaction probability (shape: [B,])
        """
        logit = self.score(user_idx, item_idx)
        prob = torch.sigmoid(logit)
        return prob

    def score(self, user_idx, item_idx):
        pred_vector = self.ensemble(user_idx, item_idx)
        logit = self.pred_layer(pred_vector).squeeze(-1)
        return logit

    def ensemble(self, user_idx, item_idx):
        # modules
        pred_vector_matching = self.MaFL.ncf(user_idx, item_idx)
        pred_vector_metric = self.MeFL.ncf(user_idx, item_idx)

        # agg
        kwargs = dict(
            tensors=(pred_vector_matching, pred_vector_metric), 
            dim=-1,
        )
        pred_vector = torch.cat(**kwargs)

        return pred_vector

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
        self.pred_layer = nn.Linear(**kwargs)