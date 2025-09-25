import torch
import torch.nn as nn

from model.BasedModel import BaseModel
from utils.tool_functions import get_dcp_tensor
import torchvision.transforms.functional as tv_tf
import torch.nn.functional as F


class DCP(BaseModel):
    def __init__(self,
                 window_size=15,
                 w=0.95,
                 ):
        super().__init__()

        self.window_size = window_size
        self.w = w

        self.frozen_module_list = nn.ModuleList([])
        self.trained_module_list = nn.ModuleList([])
        self.param_module_list = nn.ModuleList([])

        self.refine_eps = 1e-3
        self.refine_r = 31
        self.refine_avg_pool = nn.AvgPool2d(self.refine_r, stride=1, padding=self.refine_r // 2, count_include_pad=False)

        self.device = "cpu"

    def __get_a__(self, x: torch.Tensor, dc: torch.Tensor):
        b, c, h, w = x.shape
        hw = h * w

        x: torch.Tensor = x.flatten(-2, -1).permute(0, 2, 1)  # [B, HW, C]
        dc = dc.flatten(-2, -1).permute(0, 2, 1)  # [B, HW, 1]

        _, top_idx = torch.topk(dc, k=int(max(hw * 1e-3, 1)), dim=1)  # [B, k, 1]

        top_idx = top_idx.expand(-1, -1, x.size(2))

        top_pixels = x.gather(index=top_idx, dim=1)  # [B, k, C]

        intensity = top_pixels.mean(dim=-1)  # [B, k]

        top_arg = torch.argmax(intensity, dim=-1)  # [B,]

        top_arg = top_arg.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, c)

        top_pixel = top_pixels.gather(index=top_arg, dim=1)  # [B, 1, C]

        atm_light = top_pixel.permute(0, 2, 1).unsqueeze(-1)  # [B, C, 1, 1]

        return atm_light

    def guided_filter_refine(self, img, t):
        img = tv_tf.rgb_to_grayscale(img)

        mean_img = self.refine_avg_pool(img)
        mean_t = self.refine_avg_pool(t)
        mean_img_t = self.refine_avg_pool(t * img)
        cov_img_t = mean_img_t - mean_img * mean_t

        mean_img_img = self.refine_avg_pool(img * img)
        var_img = mean_img_img - mean_img * mean_img

        a = cov_img_t / (var_img + self.refine_eps)
        b = mean_t - a * mean_img

        mean_a = self.refine_avg_pool(a)
        mean_b = self.refine_avg_pool(b)

        q = mean_a * img + mean_b

        return q

    def forward(self, x):
        # [B, C, H, W]

        B, C, H, W = x.shape

        HW = H * W

        dc = get_dcp_tensor(x, s=self.window_size)

        atm_light = self.__get_a__(x, dc)

        tran = 1 - self.w * get_dcp_tensor(x / (atm_light + 1e-6), s=self.window_size)

        tran = self.guided_filter_refine(x, tran)

        tran[tran < 0.1] = 0.1

        de_smoke_x = (x - atm_light) / tran + atm_light

        pre_smoke_mask = 1 - tran

        return pre_smoke_mask, de_smoke_x

    def forward_dict(self, batch_dict):
        # [B, T, C, H, W]
        x = batch_dict["x"].to(self.device)

        pre_smoke_mask, de_smoke_x = self.forward(x)
        batch_dict["pre_gt"] = de_smoke_x.detach()

        batch_dict["vis_res"] = {}
        batch_dict["vis_res"]["pre_smoke"] = pre_smoke_mask.detach()

        return pre_smoke_mask, de_smoke_x

    def get_loss_dict(self, batch_dict, only_dict=False):
        loss_dict = {
            "Loss": 0.0,
        }

        if only_dict:
            return loss_dict

        loss_dict["Loss"] = 0

        return loss_dict

