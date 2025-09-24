class EarlyStopper:
    def __init__(
        self,
        patience: int,
        min_delta: float,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.best_epoch = 0
        self.best_score = -float("inf")
        self.best_model_state = None
        self.counter = 0
        self.stop = False

    def check(
        self,
        current_score,
        current_epoch,
        current_model_state,
    ):
        if current_score  > self.best_score + self.min_delta:
            self.best_score = current_score
            self.best_epoch = current_epoch + 1
            self.best_model_state = current_model_state
            self.counter = 0
        else:
            self.counter += 1

        if self.counter > self.patience:
            self.stop = True

    @property
    def should_stop(self):
        return self.stop
