class EarlyStopper:
    def __init__(
        self,
        patience: int,
        delta: float,
    ):
        self.patience = patience
        self.delta = delta

        self._set_up_components()

    def __call__(
        self,
        current_score,
        current_epoch,
        current_model_state,
    ):
        if current_score > self._best_score + self.delta:
            self._best_score = current_score
            self._best_epoch = current_epoch + 1
            self._best_model_state = current_model_state
            self._counter = 0
        else:
            self._counter += 1

        if self._counter > self.patience:
            self._stop = True

    @property
    def should_stop(self):
        return self._stop

    @should_stop.setter
    def should_stop(self, value):
        self._stop = bool(value)

    @property
    def get_counter(self):
        return self._counter
    
    @get_counter.setter
    def set_counter(self, value):
        self._counter = value

    @property
    def get_best_epoch(self):
        return self._best_epoch

    @property
    def get_best_score(self):
        return self._best_score

    @property
    def get_best_model_state(self):
        return self._best_model_state

    def _set_up_components(self):
        self._best_epoch = 0
        self._best_score = -float("inf")
        self._best_model_state = None
        self._counter = 0
        self._stop = False