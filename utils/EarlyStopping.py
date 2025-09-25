import numpy as np


class EarlyStopping:
    def __init__(self, epoch_patience, stop_count=0, best_value=None, lower_better=True):
        self.epoch_patience = epoch_patience
        self.stop_count = stop_count
        self.lower_better = lower_better
        if best_value is not None:
            self.best_value = best_value
        else:
            if self.lower_better:
                self.best_value = np.inf
            else:
                self.best_value = -np.inf

        self.early_stop_check = self.__early_stop_check_loss__ if self.lower_better else self.__early_stop_check_psnr__

    def __early_stop_check_loss__(self, loss):
        if loss > self.best_value:
            self.stop_count += 1
        else:
            self.best_value = loss
            self.stop_count = 0

        return self.stop_count >= self.epoch_patience

    def __early_stop_check_psnr__(self, psnr):
        if psnr < self.best_value:
            self.stop_count += 1
        else:
            self.best_value = psnr
            self.stop_count = 0

        return self.stop_count >= self.epoch_patience
