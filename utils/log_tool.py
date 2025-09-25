import numpy as np
from torch.utils.tensorboard import SummaryWriter
import tqdm


class TrainValLogger:
    def __init__(self, loss_dict: dict, metric_dict: dict):
        self.loss_dict = loss_dict.copy()
        self.metric_dict = metric_dict.copy()

        loss_title_list = list(self.loss_dict.keys())
        metric_title_list = list(self.metric_dict.keys())

        assert "Loss" in loss_title_list
        if "Loss" in loss_title_list:
            loss_title_list.remove("Loss")
        if "Grad" in loss_title_list:
            loss_title_list.remove("Grad")

        loss_title_list = ["Loss"] + sorted(loss_title_list) + sorted(metric_title_list)

        self.title_list = ["Epoch"] + loss_title_list

        self.max_length = max(map(lambda x: len(x), self.title_list))
        self.max_length = max(self.max_length, 15)

    def print_title(self, loss=True, verbose=True):
        string_list = [f"{t:^{self.max_length}}" for t in self.title_list[1:]]
        string_list = [f"{self.title_list[0]:^6}", f"{('Min_Loss' if loss else 'Min_RMSE'):^10}"] + string_list
        string = " | ".join(string_list)
        if verbose:
            print(string)
        return string

    def print_loss_metric(self, epoch, min_loss, tr_loss_dict, va_loss_dict, tr_metric_dict, va_metric_dict,
                          log_writer: SummaryWriter = None, verbose=True, saved_mark=False):
        tr_dict = tr_loss_dict | tr_metric_dict
        va_dict = va_loss_dict | va_metric_dict
        epoch_str = str(epoch) + ('' if not saved_mark else '*')
        string_list = [f"{epoch_str:^6}", f"{min_loss:^10.03f}"]
        for k in self.title_list[1:]:
            tr_v = tr_dict[k]
            va_v = va_dict[k]
            string_i = f"{tr_v:.03f} / {va_v:.03f}"
            string_list.append(f"{string_i:^{self.max_length}}")

            if log_writer is not None:
                if k in self.loss_dict:
                    log_writer.add_scalars(f"Loss", {"tr": tr_v, "va": va_v}, global_step=epoch)
                else:
                    log_writer.add_scalars(f"Metric", {"tr": tr_v, "va": va_v}, global_step=epoch)

        string = " | ".join(string_list)
        # print(string)
        if verbose:
            tqdm.tqdm.write(string)

        if log_writer is not None:
            log_writer.flush()

        return string
