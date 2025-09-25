import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
# import skvideo.measure as skvm
from piq import BRISQUELoss, SSIMLoss, CLIPIQA
import torchvision.transforms.functional as tv_tf

from utils.metric.CIEDE2000 import CIEDE2000
from utils.metric.FADE_torch import FADE
# from brisque import BRISQUE
from utils.metric.niqe_torch import NIQE


class MetricRecorder:

    def __init__(self, quick_metric, gt_mark=True):

        self.quick_metric = quick_metric
        self.gt_mark = gt_mark

        self.metric_record_dict = {
            "PSNR": [],
            "SSIM": [],
            "RMSE": [],
            "CLIPIQA": [],
            "Brisque": [],
            "NIQE": [],
            "FADE": [],
            "CIEDE2000": [],
        }

        self.brisque_obj = BRISQUELoss(reduction="none")
        self.clip_iqa_obj = CLIPIQA()
        self.niqe_obj = NIQE()
        self.ssim_obj = SSIMLoss(reduction="none")
        self.fade_obj = FADE()
        self.ciede_2000_obj = CIEDE2000()

        if self.gt_mark:
            pass
        else:
            self.metric_record_dict.pop("PSNR")
            self.metric_record_dict.pop("SSIM")
            self.metric_record_dict.pop("RMSE")
            self.metric_record_dict.pop("CIEDE2000")

            self.ssim_obj = None

        if self.quick_metric:
            if self.metric_record_dict.get("SSIM") is not None:
                self.metric_record_dict.pop("SSIM")
            if self.metric_record_dict.get("CIEDE2000") is not None:
                self.metric_record_dict.pop("CIEDE2000")
            self.metric_record_dict.pop("CLIPIQA")
            self.metric_record_dict.pop("NIQE")
            self.metric_record_dict.pop("Brisque")
            self.metric_record_dict.pop("FADE")

            self.ssim_obj = None
            self.clip_iqa_obj = None
            self.niqe_obj = None
            self.brisque_obj = None
            self.fade_obj = None
            self.ciede_2000_obj = None

        self.metric_record_dict_dummy = self.metric_record_dict.copy()
        self.metric_record_dict["sample_name"] = []
        self.metric_record_dict["Time Cost"] = []
        self.metric_record_dict["FPS"] = []

    def record_metric(self, batch_dict: dict):
        pre_gt = batch_dict["pre_gt"]
        device = pre_gt.device

        if self.gt_mark:
            gt = batch_dict["gt"]
            gt = gt.to(device)
        else:
            gt = None

        time_cost = batch_dict["time_cost"]
        sample_name_list = batch_dict["sample_name"]

        if self.clip_iqa_obj is not None:
            self.clip_iqa_obj.to(device)

        if self.gt_mark:
            gt = torch.clamp(gt, 0, 1)

        pre_gt = torch.clamp(pre_gt, 0, 1)

        if len(pre_gt.shape) >= 5:
            if self.gt_mark:
                gt = gt[:, -1, ...]
            pre_gt = pre_gt[:, -1, ...]
            sample_name_list = sample_name_list[-1]

        n = len(pre_gt)

        if self.brisque_obj is not None:
            brisque_all = self.brisque_obj(pre_gt).squeeze().tolist()
        else:
            brisque_all = []

        if self.clip_iqa_obj is not None:
            clip_iqa_all = self.clip_iqa_obj(pre_gt).squeeze().tolist()
        else:
            clip_iqa_all = []

        if self.niqe_obj is not None:
            pre_gt_gray = tv_tf.rgb_to_grayscale(pre_gt)[:, 0, :, :] * 255
            niqe_all = self.niqe_obj.score(pre_gt_gray).squeeze().tolist()
        else:
            niqe_all = []

        if self.fade_obj is not None:
            fade_all = self.fade_obj.score(pre_gt * 255).squeeze().tolist()
        else:
            fade_all = []

        if self.gt_mark:
            mse_all = F.mse_loss(pre_gt, gt, reduction="none").mean(dim=[1, 2, 3], keepdim=True)
            rmse_all = mse_all.sqrt()
            psnr_all = - 10 * torch.log10(mse_all + 1e-10)
            psnr_all = psnr_all.squeeze().tolist()
            rmse_all = rmse_all.squeeze().tolist()
            mse_all = mse_all.squeeze().tolist()
            if self.ciede_2000_obj is not None:
                ciede_2000_all = self.ciede_2000_obj.score(pre_gt * 255, gt * 255).squeeze().tolist()
            else:
                ciede_2000_all = []
            if self.ssim_obj is not None:
                ssim_all = (1 - self.ssim_obj(pre_gt, gt)).squeeze().tolist()
            else:
                ssim_all = []
        else:
            psnr_all = []
            rmse_all = []
            mse_all = []
            ssim_all = []
            ciede_2000_all = []

        if n == 1:
            if brisque_all is not None:
                brisque_all = [brisque_all]
            if clip_iqa_all is not None:
                clip_iqa_all = [clip_iqa_all]
            if niqe_all is not None:
                niqe_all = [niqe_all]
            if fade_all is not None:
                fade_all = [fade_all]

            if self.gt_mark:
                rmse_all = [rmse_all]
                mse_all = [mse_all]
                psnr_all = [psnr_all]
                if ciede_2000_all is not None:
                    ciede_2000_all = [ciede_2000_all]
                if ssim_all is not None:
                    ssim_all = [ssim_all]

        time_cost_i = time_cost / n

        sample_name_list = list(sample_name_list)
        time_cost_list = [time_cost_i] * n

        self.metric_record_dict["Time Cost"] += time_cost_list
        self.metric_record_dict["sample_name"] += sample_name_list

        if self.metric_record_dict.get("Brisque") is not None:
            self.metric_record_dict["Brisque"] += brisque_all
        if self.metric_record_dict.get("CLIPIQA") is not None:
            self.metric_record_dict["CLIPIQA"] += clip_iqa_all
        if self.metric_record_dict.get("NIQE") is not None:
            self.metric_record_dict["NIQE"] += niqe_all
        if self.metric_record_dict.get("FADE") is not None:
            self.metric_record_dict["FADE"] += fade_all

        if self.metric_record_dict.get("PSNR") is not None:
            self.metric_record_dict["PSNR"] += psnr_all
        if self.metric_record_dict.get("RMSE") is not None:
            self.metric_record_dict["RMSE"] += rmse_all
        if self.metric_record_dict.get("SSIM") is not None:
            self.metric_record_dict["SSIM"] += ssim_all
        if self.metric_record_dict.get("CIEDE2000") is not None:
            self.metric_record_dict["CIEDE2000"] += ciede_2000_all

        return

    def get_metric_dict(self):
        metric_dict = {}

        n = len(self.metric_record_dict["sample_name"])

        if self.metric_record_dict.get("PSNR") is not None:
            metric_dict["PSNR"] = sum(self.metric_record_dict["PSNR"]) / n
        if self.metric_record_dict.get("RMSE") is not None:
            metric_dict["RMSE"] = sum(self.metric_record_dict["RMSE"]) / n
        if self.metric_record_dict.get("CIEDE2000") is not None:
            metric_dict["CIEDE2000"] = sum(self.metric_record_dict["CIEDE2000"]) / n
        if self.metric_record_dict.get("SSIM") is not None:
            metric_dict["SSIM"] = sum(self.metric_record_dict["SSIM"]) / n
        if self.metric_record_dict.get("CLIPIQA") is not None:
            metric_dict["CLIPIQA"] = sum(self.metric_record_dict["CLIPIQA"]) / n
        if self.metric_record_dict.get("Brisque") is not None:
            metric_dict["Brisque"] = sum(self.metric_record_dict["Brisque"]) / n
        if self.metric_record_dict.get("NIQE") is not None:
            metric_dict["NIQE"] = sum(self.metric_record_dict["NIQE"]) / n
        if self.metric_record_dict.get("FADE") is not None:
            metric_dict["FADE"] = sum(self.metric_record_dict["FADE"]) / n

        total_time_cost = sum(self.metric_record_dict["Time Cost"])

        metric_dict["FPS"] = n / total_time_cost if total_time_cost else 0
        metric_dict["Time Cost"] = total_time_cost / n

        return metric_dict

    def save_metric(self, metric_dict, output_dir):
        dataframe = pd.Series(metric_dict)
        dataframe.to_excel(f"{output_dir}/metric.xlsx")

        metric_detail = self.metric_record_dict.copy()
        metric_detail.pop("FPS")

        column_list = list(metric_detail.keys())
        column_list.remove("sample_name")
        column_list = ["sample_name"] + column_list
        dataframe = pd.DataFrame(metric_detail)
        dataframe.to_excel(f"{output_dir}/metric_details.xlsx", index=False, columns=column_list)
        return


def print_metric(metric_dict):
    for k in metric_dict:
        print(f"{k}: {metric_dict[k]}")


def log_metric(metric_dict, log_writer: SummaryWriter):
    for k in metric_dict:
        log_writer.add_scalar(
            f"Test Metric/{k}",
            metric_dict[k],
            global_step=0,
        )


def save_metric(metric_dict, output_dir):
    dataframe = pd.Series(metric_dict)
    dataframe.to_excel(f"{output_dir}/metric.xlsx")
