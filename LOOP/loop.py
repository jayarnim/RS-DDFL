from IPython.display import clear_output
from statistics import mean
import torch


class TrainingLoop:
    def __init__(
        self, 
        model, 
        trainer,
        monitor,
    ):
        # device setting
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(DEVICE)

        # global attr
        self.model = model.to(self.device)
        self.trainer = trainer
        self.monitor = monitor

    def fit(
        self, 
        trn_loader: torch.utils.data.dataloader.DataLoader, 
        val_loader: torch.utils.data.dataloader.DataLoader, 
        loo_loader: torch.utils.data.dataloader.DataLoader, 
        n_epochs: int, 
        warm_up: int=10,
        interval: int=1,
    ):
        trn_task_loss_list = []
        val_task_loss_list = []
        computing_cost_list = []

        for epoch in range(n_epochs):
            if epoch % 10 == 0:
                print(f"EPOCH {epoch+1} START ---->>>>")

            # trn, val
            kwargs = dict(
                trn_loader=trn_loader, 
                val_loader=val_loader, 
                epoch=epoch,
                n_epochs=n_epochs,
            )
            trn_task_loss, val_task_loss, computing_cost_list_per_batch = self.trainer.fit(**kwargs)

            trn_task_loss_list.append(trn_task_loss)
            val_task_loss_list.append(val_task_loss)
            computing_cost_list.extend(computing_cost_list_per_batch)

            print(
                f"TRN TASK LOSS: {trn_task_loss:.4f}",
                f"VAL TASK LOSS: {val_task_loss:.4f}",
                sep='\n'
            )

            # early stopping
            if (epoch+1 > warm_up) and ((epoch+1) % interval == 0):
                kwargs = dict(
                    dataloader=loo_loader, 
                    epoch=epoch,
                )
                current_score = self.monitor.monitor(**kwargs)

                if self.monitor.stopper.should_stop:
                    break
                else:
                    print(
                        f"CURRENT SCORE: {current_score:.4f}",
                        f"BEST SCORE: {self.monitor.stopper.best_score:.4f}",
                        f"BEST EPOCH: {self.monitor.stopper.best_epoch}",
                        sep='\t',
                    )

            # log reset
            if (epoch + 1) % 50 == 0:
                clear_output(wait=False)

        history = dict(
            trn=trn_task_loss_list,
            val=val_task_loss_list,
        )

        best_epoch = self.monitor.stopper.best_epoch
        best_score = self.monitor.stopper.best_score
        best_model_state = self.monitor.stopper.best_model_state

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        clear_output(wait=False)

        print(
            "LEAVE ONE OUT",
            f"\tBEST SCORE: {best_score:.4f}",
            f"\tBEST EPOCH: {best_epoch}",
            sep="\n",
        )
        print(
            "COMPUTING COST FOR LEARNING",
            f"\t(s/epoch): {sum(computing_cost_list)/n_epochs:.4f}",
            f"\t(epoch/s): {n_epochs/sum(computing_cost_list):.4f}",
            f"\t(s/batch): {mean(computing_cost_list):.4f}",
            f"\t(batch/s): {1.0/mean(computing_cost_list):.4f}",
            sep="\n",
        )

        return history