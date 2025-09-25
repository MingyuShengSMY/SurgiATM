import argparse
import os
import time
import numpy as np
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.cuda
from model import BasedModel
from torch.utils.tensorboard import SummaryWriter
from dataloader import get_dataset, get_dataloader
from utils.Config import Config, load_config
from utils.EarlyStopping import EarlyStopping
from utils.Metric import MetricRecorder, print_metric, log_metric, save_metric
from utils.log_tool import TrainValLogger
from utils.model_selection import select_model, select_opt
from utils.tool_functions import count_model_parameters, seed_everything, wait
import tqdm
import shutil
import datetime
from utils.visualizer import visualize_dict

NO_MODEL_METHOD_LIST = ["DCP"]


class Method:
    def __init__(self, config, vis=False):

        self.run_check = False  # to check bug
        # self.run_check = True  # to check bug, skip most loop, e.g. train, epoch loop

        self.one_sample_overfitting = False  # to check whether the model works or not
        # self.one_sample_overfitting = True  # to check whether the model works or not

        self.quick_metric = True  # Some metrics are too slow to compute when training. Skip them
        # self.quick_metric = False

        self.config = config
        self.method_name = self.config.method.name
        self.opt_config = config.method.opt
        self.output_dir = config.output.dir
        self.output_result_dir = f"{self.output_dir}/result"
        self.saved_model_dir = f"{self.output_dir}/saved_model"
        self.saved_model_path = f"{self.saved_model_dir}/model.pt"
        self.model_load_dir = f"{config.model_load_dir}/saved_model/model.pt" if config.model_load_dir is not None else self.saved_model_path
        self.log_dir = config.log.dir
        self.log_txt_path = f"{self.log_dir}/log.txt"
        self.device = "cuda" if torch.cuda.is_available() and self.config.use_gpu else "cpu"
        self.verbose = self.config.verbose

        seed_everything(self.config.random_seed)

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.saved_model_dir, exist_ok=True)

        self.model = select_model(self.method_name, config.method).to(self.device)
        self.model.device = self.device
        self.opts = self.model.config_opt(self.opt_config[0])
        self.loss_dict_dummy = self.model.get_loss_dict(None, only_dict=True)  # for printing the table column title
        self.loss_dict_dummy = {k: 0 for k in self.loss_dict_dummy}

        self.model_param_count = count_model_parameters(self.model.param_module_list)
        self.model_flops = self.model.get_flops()
        self.model_trainable_param_count = count_model_parameters(self.model.trained_module_list)

        self.tr_dataset, self.va_dataset, self.te_dataset = get_dataset(self.config, self.one_sample_overfitting, vis=vis)
        self.tr_loader, self.va_loader, self.te_loader = get_dataloader(self.config, self.tr_dataset, self.va_dataset,
                                                                        self.te_dataset, vis=vis)
        self.gt_mark = self.config.dataset.gt_mark

        self.save_metric = self.config.training.save_metric

        print(f"Method Name: {self.method_name}")
        print(f"Param Count: {self.model_param_count * 1e-6:.02f} M")
        print(f"FLOPs: {self.model_flops * 1e-9:.02f} B")
        print(f"Trainable Param Count: {self.model_trainable_param_count * 1e-6:.02f} M")
        print(f"Dataset Name: {self.config.dataset.name}")

    def __train_val_iter__(self, epoch, data_loader, train=True, vis=False, metric_recorder: MetricRecorder = None):
        if train:
            self.model.train()
            self.model.requires_grad_(True)
            self.model.frozen_module_list.eval()
            self.model.frozen_module_list.requires_grad_(False)
        else:
            self.model.eval()
            self.model.requires_grad_(False)

        batch_range = tqdm.tqdm(
            data_loader,
            position=1,
            desc="Training" if train else "Validating",
            leave=False
        )

        batch_count = 0

        loss_dict = self.loss_dict_dummy.copy()

        batch_dict = {}

        for batch_dict in batch_range:
            if train:
                wait()

            batch_dict["time_cost"] = 0

            if train and len(batch_dict["x"]) == 1 and not self.one_sample_overfitting:
                continue

            batch_count += 1

            loss_dict_i = self.model.get_loss_dict(batch_dict)

            if train:
                self.model.opt_update(loss_dict_i, opts=self.opts)

            loss_dict = {
                k: loss_dict[k] + (loss_dict_i[k].item() if torch.is_tensor(loss_dict_i[k]) else loss_dict_i[k]) for k
                in loss_dict}

            metric_recorder.record_metric(batch_dict)  # ssim is too slow

            if self.run_check:
                break

        if vis and len(batch_dict) and not train:
            if not train:
                output_dir = f"{self.output_dir}/train_vis/val"
            else:
                output_dir = f"{self.output_dir}/train_vis/tra"

            batch_dict_vis = batch_dict.copy()

            visualize_dict(batch_dict_vis, output_dir=output_dir, fixed_sample_name=f"{epoch:04d}.png")

        loss_dict = {k: loss_dict[k] / batch_count for k in loss_dict}
        loss_dict = {k: loss_dict[k].item() if torch.is_tensor(loss_dict[k]) else loss_dict[k] for k in loss_dict}

        metric_dict = metric_recorder.get_metric_dict()

        return loss_dict, metric_dict

    def train(self, vis=False):

        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
        os.makedirs(self.log_dir, exist_ok=True)

        if self.one_sample_overfitting:
            log_writer = None
        else:
            log_writer = SummaryWriter(log_dir=self.log_dir)

        min_va_loss = np.inf
        min_va_rmse = np.inf

        metric_recorder_dummy = MetricRecorder(quick_metric=self.quick_metric, gt_mark=self.gt_mark)
        metric_dict_dummy = metric_recorder_dummy.metric_record_dict_dummy.copy()

        train_va_logger = TrainValLogger(self.loss_dict_dummy, metric_dict_dummy)
        log_string = train_va_logger.print_title(loss=self.save_metric=="Loss", verbose=self.verbose)

        with open(self.log_txt_path, "w") as f:
            f.write(log_string + "\n")

        max_epoch = self.config.training.epoch if not self.one_sample_overfitting else 1000000

        early_stop_checker_loss = EarlyStopping(epoch_patience=max(5, self.config.training.patience), lower_better=True)
        early_stop_checker_rmse = EarlyStopping(epoch_patience=max(5, self.config.training.patience), lower_better=True)

        epoch_tqdm = tqdm.tqdm(
            range(max_epoch),
            leave=False,
            position=0,
            desc="Epoch"
        )

        for epoch in epoch_tqdm:
            # epoch_tqdm.set_description(f"Epoch {epoch:03d}")

            self.model.before_epoch(epoch, max_epoch)

            metric_recorder_tr = MetricRecorder(quick_metric=self.quick_metric, gt_mark=self.gt_mark)
            metric_recorder_va = MetricRecorder(quick_metric=self.quick_metric, gt_mark=self.gt_mark)

            tr_loss_dict, tr_metric_dict = self.__train_val_iter__(epoch, self.tr_loader, train=True, vis=vis, metric_recorder=metric_recorder_tr)
            va_loss_dict, va_metric_dict = self.__train_val_iter__(epoch, self.va_loader, train=False, vis=vis, metric_recorder=metric_recorder_va)

            va_loss = va_loss_dict["Loss"]
            va_rmse = va_metric_dict["RMSE"]

            saved_mark = False
            if (self.save_metric == "Loss" and va_loss < min_va_loss) or va_rmse < min_va_rmse:
                if va_loss < min_va_loss:
                    min_va_loss = va_loss

                if va_rmse < min_va_rmse:
                    min_va_rmse = va_rmse
                    if not self.one_sample_overfitting:
                        torch.save(
                            obj=self.model,
                            f=self.saved_model_path
                        )
                        saved_mark = True

            log_string = train_va_logger.print_loss_metric(epoch, min_va_loss if self.save_metric == "Loss" else min_va_rmse,
                                                   tr_loss_dict, va_loss_dict,
                                                   tr_metric_dict, va_metric_dict,
                                                   log_writer=log_writer, verbose=self.verbose, saved_mark=saved_mark)
            with open(self.log_txt_path, "a") as f:
                f.write(log_string + "\n")

            loss_early_stop = early_stop_checker_loss.early_stop_check(va_loss)
            rmse_early_stop = early_stop_checker_rmse.early_stop_check(va_rmse)

            if not self.one_sample_overfitting and rmse_early_stop or np.isnan(va_loss):
                tqdm.tqdm.write("Early Stopped")
                break

            if self.run_check and max_epoch > 2:
                break

        if self.one_sample_overfitting or self.run_check:
            torch.save(
                obj=self.model,
                f=self.saved_model_path
            )

        if log_writer is not None:
            log_writer.close()

    def __test__(self, dataloader, metric_recorder, vis=False):
        self.model.eval()
        self.model.requires_grad_(False)

        batch_range = tqdm.tqdm(
            self.te_loader,
            position=0,
            desc="Testing",
            leave=False
        )

        batch_count = 0

        for batch_dict in batch_range:
            batch_count += 1

            s_t = time.time()
            self.model.forward_dict(batch_dict)
            e_t = time.time()
            batch_dict["time_cost"] = e_t - s_t

            metric_recorder.record_metric(batch_dict)

            if vis:
                visualize_dict(batch_dict, output_dir=self.output_result_dir)

            if self.run_check:
                break

        return

    def test(self, vis=False):
        if self.method_name not in NO_MODEL_METHOD_LIST:
            self.model: BasedModel = torch.load(self.model_load_dir, weights_only=False)

        metric_recorder = MetricRecorder(quick_metric=False, gt_mark=self.gt_mark)

        self.__test__(self.te_loader, metric_recorder=metric_recorder, vis=vis)

        metric_dict = metric_recorder.get_metric_dict()

        metric_dict["Param"] = self.model_param_count * 1e-6
        metric_dict["FLOPs"] = self.model_flops * 1e-9

        print_metric(metric_dict)

        metric_recorder.save_metric(metric_dict, output_dir=self.output_dir)

        return metric_dict


def main():
    args = get_args()
    # args = parser.parse_args()

    if not args.train and not args.test:
        raise ValueError("'--train' or '--test'?")

    print()
    print("==========================================================================================")
    print()

    print(f"Loading Config From: {args.config_file}")

    config = load_config(args.config_file)

    config = Config(config)

    method = Method(config, args.vis)

    if args.train and not config.training.no_train_mark:
        print()
        date_time = datetime.datetime.now().strftime("%H:%M:%S on %Y/%m/%d")
        print(f"Training Start at {date_time}")
        print()

        method.train(args.vis)

        print()
        date_time = datetime.datetime.now().strftime("%H:%M:%S on %Y/%m/%d")
        print(f"Training Done at {date_time}")

    if args.test:
        print()
        date_time = datetime.datetime.now().strftime("%H:%M:%S on %Y/%m/%d")
        print(f"Testing Start at {date_time}")
        print()

        method.test(args.vis)

        print()
        date_time = datetime.datetime.now().strftime("%H:%M:%S on %Y/%m/%d")
        print(f"Testing Done at {date_time}")
        print()


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config_file', required=True, help='path/to/target/config_file.yaml')
    parser.add_argument('--train', default=False, action="store_true", help='train mode')
    parser.add_argument('--test', default=False, action="store_true", help='test mode')
    parser.add_argument('--vis', default=False, action="store_true", help='vis mode')

    return parser.parse_args()


if __name__ == '__main__':
    main()
