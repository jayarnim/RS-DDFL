import torch
import torch.nn as nn


class Module(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int,
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
        self.hidden = hidden
        self.dropout = dropout
        self.register_buffer(
            name="interactions", 
            tensor=interactions,
        )

        # debugging args error
        self._assert_arg_error()

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
        pred_vector = self.ml(user_idx, item_idx)
        logit = self.logit_layer(pred_vector).squeeze(-1)
        return logit

    def ml(self, user_idx, item_idx):
        proj_user = self.user(user_idx, item_idx)
        proj_item = self.item(user_idx, item_idx)

        kwargs = dict(
            tensors=(proj_user, proj_item), 
            dim=-1,
        )
        concat = torch.cat(**kwargs)
        pred_vector = self.mlp_layers(concat)

        return pred_vector

    def user(self, user_idx, item_idx):
        # get user vector from interactions
        user_slice = self.interactions[user_idx, :-1].clone()
        
        # masking target items
        user_batch = torch.arange(user_idx.size(0))
        user_slice[user_batch, item_idx] = 0
        
        # projection
        proj_user = self.proj_u(user_slice.float())

        return proj_user

    def item(self, user_idx, item_idx):
        # get item vector from interactions
        item_slice = self.interactions.T[item_idx, :-1].clone()
        
        # masking target users
        item_batch = torch.arange(item_idx.size(0))
        item_slice[item_batch, user_idx] = 0
        
        # projection
        proj_item = self.proj_i(item_slice.float())

        return proj_item

    def _init_layers(self):
        kwargs = dict(
            in_features=self.n_items,
            out_features=self.n_factors,
            bias=False,
        )
        self.proj_u = nn.Linear(**kwargs)

        kwargs = dict(
            in_features=self.n_users,
            out_features=self.n_factors,
            bias=False,
        )
        self.proj_i = nn.Linear(**kwargs)

        self.mlp_layers = nn.Sequential(
            *list(self._generate_layers(self.hidden))
        )

        kwargs = dict(
            in_features=self.hidden[-1],
            out_features=1,
        )
        self.logit_layer = nn.Linear(**kwargs)

    def _generate_layers(self, hidden):
        idx = 1
        while idx < len(hidden):
            yield nn.Linear(hidden[idx-1], hidden[idx])
            yield nn.LayerNorm(hidden[idx])
            yield nn.ReLU()
            yield nn.Dropout(self.dropout)
            idx += 1

    def _assert_arg_error(self):
        CONDITION = (self.hidden[0] == self.n_factors * 2)
        ERROR_MESSAGE = f"First MLP layer must match input size: {self.n_factors * 2}"
        assert CONDITION, ERROR_MESSAGE