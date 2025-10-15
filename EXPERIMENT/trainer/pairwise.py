from tqdm import tqdm
from time import perf_counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from ..utils.constants import LOSS_FN_TYPE_PAIRWISE
from ..loss_fn import pairwise
from PIPELINE.dataloader.pairwise import CustomizedDataLoader


class CustomizedTrainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn_type: LOSS_FN_TYPE_PAIRWISE="bpr",
        lr: float=1e-4, 
        weight_decay: float=1e-3, 
    ):
        """
        Pairwise Learning Single Epoch Trainer for Latent Factor Model
        -----
        created by @jayarnim

        Args:
            model (nn.Module):
                latent factor model instance.
            loss_fn_type (str):
                parwise loss functions currently supported are: `bpr`.
            lr (float):
                learning rate of optimizer `adam`.
            weight_decay (float):
                L2 regularization strength of optimizer `adam`.
        """
        # device setting
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(DEVICE)

        # global attr
        self.model = model.to(self.device)
        self.loss_fn_type = loss_fn_type
        self.lr = lr
        self.weight_decay = weight_decay
        
        # set up components, loss function, optimizer, etc.
        self._set_up_components()

    def __call__(
        self, 
        trn_loader: CustomizedDataLoader, 
        val_loader: CustomizedDataLoader, 
        epoch: int,
        n_epochs: int,
    ):
        kwargs = dict(
            dataloader=trn_loader,
            epoch=epoch,
            n_epochs=n_epochs,
        )
        trn_loss, computing_cost = self._epoch_trn_step(**kwargs)

        kwargs = dict(
            dataloader=val_loader,
            epoch=epoch,
            n_epochs=n_epochs,
        )
        val_loss = self._epoch_val_step(**kwargs)

        return trn_loss, val_loss, computing_cost

    def _epoch_trn_step(
        self,
        dataloader: torch.utils.data.dataloader.DataLoader,
        epoch: int,
        n_epochs: int,
    ):
        self.model.train()

        epoch_loss = 0.0
        epoch_computing_cost = []

        iter_obj = tqdm(
            iterable=dataloader, 
            desc=f"Epoch {epoch+1}/{n_epochs} TRN"
        )

        for user_idx, pos_idx, neg_idx in iter_obj:
            # to gpu
            kwargs = dict(
                user_idx=user_idx.to(self.device),
                pos_idx=pos_idx.to(self.device), 
                neg_idx=neg_idx.to(self.device),
            )

            # set starting time for computing cost
            t0 = perf_counter()

            # forward pass
            with autocast(self.device.type):
                batch_loss = self._batch_step(**kwargs)

            # backward pass
            self._run_fn_opt(batch_loss)

            # calculate computing cost
            batch_computing_cost = perf_counter() - t0

            # accumulate loss
            epoch_loss += batch_loss.item()
            epoch_computing_cost.append(batch_computing_cost)

        return epoch_loss / len(dataloader), epoch_computing_cost

    @torch.no_grad()
    def _epoch_val_step(        
        self,
        dataloader: torch.utils.data.dataloader.DataLoader,
        epoch: int,
        n_epochs: int,
    ):
        self.model.eval()

        epoch_loss = 0.0

        iter_obj = tqdm(
            iterable=dataloader, 
            desc=f"Epoch {epoch+1}/{n_epochs} VAL"
        )

        for user_idx, pos_idx, neg_idx in iter_obj:
            # to gpu
            kwargs = dict(
                user_idx=user_idx.to(self.device),
                pos_idx=pos_idx.to(self.device), 
                neg_idx=neg_idx.to(self.device),
            )

            # forward pass
            with autocast(self.device.type):
                batch_loss = self._batch_step(**kwargs)

            # accumulate loss
            epoch_loss += batch_loss.item()

        return epoch_loss / len(dataloader)

    def _batch_step(self, user_idx, pos_idx, neg_idx):
        pos_logit = self.model(user_idx, pos_idx)
        neg_logit = self.model(user_idx, neg_idx)
        loss = self.loss_fn(pos_logit, neg_logit)
        return loss

    def _run_fn_opt(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def _set_up_components(self):
        self._init_loss_fn()
        self._init_optimizer()
        self._init_scaler()

    def _init_loss_fn(self):
        if self.loss_fn_type=="bpr":
            self.loss_fn = pairwise.bpr
        else:
            raise ValueError(f"Invalid loss_fn_type: {self.loss_fn_type}")

    def _init_optimizer(self):
        kwargs = dict(
            params=self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay,
        )
        self.optimizer = optim.Adam(**kwargs)

    def _init_scaler(self):
        kwargs = dict(
            device=self.device,
        )
        self.scaler = GradScaler(**kwargs)