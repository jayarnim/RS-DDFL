from tqdm import tqdm
from time import perf_counter
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast


class PairwiseTrainer:
    def __init__(
        self,
        model,
        task_fn,
        lr: float=1e-4, 
        lambda_: float=1e-3, 
    ):
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(DEVICE)

        self.model = model
        self.task_fn = task_fn
        
        # optimizer
        kwargs = dict(
            params=self.model.parameters(), 
            lr=lr, 
            weight_decay=lambda_,
        )
        self.optimizer = optim.Adam(**kwargs)

        # gradient scaler setting
        self.scaler = GradScaler()

    def fit(
        self, 
        trn_loader: torch.utils.data.dataloader.DataLoader, 
        val_loader: torch.utils.data.dataloader.DataLoader, 
        epoch: int,
        n_epochs: int,
    ):
        kwargs = dict(
            dataloader=trn_loader,
            epoch=epoch,
            n_epochs=n_epochs,
        )
        trn_task_loss, batch_computing_cost_list = self._epoch_trn_step(**kwargs)

        kwargs = dict(
            dataloader=val_loader,
            epoch=epoch,
            n_epochs=n_epochs,
        )
        val_task_loss = self._epoch_val_step(**kwargs)

        return trn_task_loss, val_task_loss, batch_computing_cost_list

    def _epoch_trn_step(
        self,
        dataloader: torch.utils.data.dataloader.DataLoader,
        epoch: int,
        n_epochs: int,
    ):
        self.model.train()

        epoch_task_loss = 0.0
        batch_computing_cost_list = []

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
                batch_task_loss = self._batch_step(**kwargs)

            # backward pass
            self._run_fn_opt(batch_task_loss)

            # calculate computing cost
            batch_computing_cost = perf_counter() - t0

            # accumulate loss
            epoch_task_loss += batch_task_loss.item()
            batch_computing_cost_list.append(batch_computing_cost)

        return epoch_task_loss / len(dataloader), batch_computing_cost_list

    def _epoch_val_step(        
        self,
        dataloader: torch.utils.data.dataloader.DataLoader,
        epoch: int,
        n_epochs: int,
    ):
        self.model.eval()

        epoch_task_loss = 0.0

        iter_obj = tqdm(
            iterable=dataloader, 
            desc=f"Epoch {epoch+1}/{n_epochs} VAL"
        )

        with torch.no_grad():
            for user_idx, pos_idx, neg_idx in iter_obj:
                # to gpu
                kwargs = dict(
                    user_idx=user_idx.to(self.device),
                    pos_idx=pos_idx.to(self.device), 
                    neg_idx=neg_idx.to(self.device),
                )

                # forward pass
                with autocast(self.device.type):
                    batch_task_loss = self._batch_step(**kwargs)

                # accumulate loss
                epoch_task_loss += batch_task_loss.item()

        return epoch_task_loss / len(dataloader)

    def _batch_step(self, user_idx, pos_idx, neg_idx):
        pos_logit = self.model(user_idx, pos_idx)
        neg_logit = self.model(user_idx, neg_idx)
        batch_task_loss = self.task_fn(pos_logit, neg_logit)
        return batch_task_loss

    def _run_fn_opt(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()