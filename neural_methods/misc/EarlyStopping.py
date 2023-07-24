import numpy as np
class EarlyStopper:
    "https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch"
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.prev_loss = np.inf
        self.counter = 0

    def early_stop(self, loss):
        if loss < self.prev_loss:
            if abs(loss - self.prev_loss) <= self.min_delta:
                if self.counter >= self.patience:
                    return True
                else:
                    self.prev_loss = loss
        else:
            if abs(loss - self.prev_loss) >= self.min_delta:
                if self.counter >= self.patience:
                    return True
                else:
                    self.prev_loss = loss
        self.counter+= 1
        return False
