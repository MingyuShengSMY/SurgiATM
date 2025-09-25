"""
Following the work:

Wang, C., Zhao, M., Zhou, C., Dong, N., Khan, Z.A., Zhao, X., Cheikh, F.A., Beghdadi, A. and Chen, S., 2024.
Smoke veil prior regularized surgical field desmoking without paired in-vivo data.
Computers in Biology and Medicine, 168, p.107761.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from model.BasedModel import BaseModel
import timm
import torchvision.transforms.functional as TF

from model.SurgiATM import SurgiATM


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        groups = 3

        self.en_block1 = nn.Sequential(
            self.build_down_conv_block(3, 64 * groups, 7, 1, 3),
            *[self.build_res_block(64 * groups, groups=groups) for _ in range(3)],
        )

        self.en_block2 = nn.Sequential(
            self.build_down_conv_block(64 * groups, 128 * groups, 5, 2, 2, groups=groups),
            *[self.build_res_block(128 * groups, groups=groups) for _ in range(3)],
        )

        self.neck_block = nn.Sequential(
            self.build_down_conv_block(128 * groups, 256 * groups, 3, 2, 1, groups=groups),
            *[self.build_res_block(256 * groups, groups=groups) for _ in range(6)],
            self.build_up_conv_block(256 * groups, 128 * groups, groups=groups),
        )

        self.de_block2 = nn.Sequential(
            *[self.build_res_block(128 * groups, groups=groups) for _ in range(3)],
            self.build_up_conv_block(128 * groups, 64 * groups, groups=groups),
        )

        self.de_block1 = nn.Sequential(
            *[self.build_res_block(64 * groups, groups=groups) for _ in range(3)],
        )

        self.output_layer_f = nn.Sequential(
            self.build_down_conv_block(64, 3, 7, 1, 3, act=False, bn=False),
            nn.Sigmoid(),
        )

        self.output_layer_t = nn.Sequential(
            self.build_down_conv_block(64, 1, 7, 1, 3, act=False, bn=False),
            nn.Sigmoid(),
        )

        self.output_layer_j = nn.Sequential(
            self.build_down_conv_block(64, 3, 7, 1, 3, act=False, bn=False),
            nn.Sigmoid(),
        )

    def build_down_conv_block(self, in_dim, out_dim, kz=3, stride=1, padding=1, act=True, bn=True, groups=1):

        block = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kz, stride, padding, groups=groups),
            nn.BatchNorm2d(out_dim) if bn else nn.Identity(),
            nn.LeakyReLU(inplace=True) if act else nn.Identity(),
        )

        return block

    def build_up_conv_block(self, in_dim, out_dim, kz=4, stride=2, padding=1, act=True, bn=True, groups=1):

        block = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kz, stride, padding, groups=groups),
            nn.BatchNorm2d(out_dim) if bn else nn.Identity(),
            nn.LeakyReLU(inplace=True) if act else nn.Identity(),
        )

        return block

    def build_res_block(self, dim, groups=1):

        block = nn.Sequential(
            self.build_down_conv_block(dim, dim, bn=False, groups=groups),
        )

        return block

    def forward(self, x):
        # [B, 3, H, W]

        en1 = self.en_block1(x)
        en2 = self.en_block2(en1)

        neck = self.neck_block(en2)

        if neck.size()[2:] != en2.size()[2:]:
            neck = F.interpolate(neck, en2.size()[2:], mode="bilinear")

        de2 = self.de_block2(neck + en2)

        if de2.size()[2:] != en1.size()[2:]:
            de2 = F.interpolate(de2, en1.size()[2:], mode="bilinear")

        de1 = self.de_block1(de2 + en1)

        de1_f, de1_j, de1_t = torch.split(de1, split_size_or_sections=64, dim=1)

        output = self.output_layer_j(de1_j)
        f = self.output_layer_f(de1_f)
        t = self.output_layer_t(de1_t)

        if output.size()[2:] != x.size()[2:]:
            output = F.interpolate(output, x.size()[2:], mode="bilinear")
            f = F.interpolate(f, x.size()[2:], mode="bilinear")
            t = F.interpolate(t, x.size()[2:], mode="bilinear")

        return output, f, t


class SVPNet(BaseModel):
    def __init__(self):
        super().__init__()

        self.net = UNet()

        self.frozen_module_list = nn.ModuleList([])
        self.trained_module_list = nn.ModuleList([self.net])
        self.param_module_list = nn.ModuleList([self.net])

        self.device = "cpu"

    def forward(self, x):
        # [B, 3, H, W]

        output, f, t = self.net(x)

        return output, f, t

    def forward_dict(self, batch_dict):
        x = batch_dict["x"].to(self.device)
        # [B, 3, H, W]

        de_smoke_x, f, t = self.forward(x)
        batch_dict["pre_gt"] = de_smoke_x.detach()

        batch_dict["vis"] = {}
        batch_dict["vis"]["F"] = f.detach()
        batch_dict["vis"]["t"] = t.detach()

        return de_smoke_x, f, t

    def get_svp_loss(self, f: torch.Tensor):

        # all_p = f.nelement()

        df_x = f.roll(1, dims=2).sub(f).square()
        df_y = f.roll(1, dims=3).sub(f).square()
        df_c = f.roll(1, dims=1).sub(f).square()

        df = df_x.add(df_y).add(df_c).sqrt().mean()

        return df

    def get_loss_dict(self, batch_dict, only_dict=False):
        loss_dict = {
            "Grad": 0,
            "Loss": 0,
            "Rec": 0,
            "Con": 0,
            "SVP": 0,
        }

        if only_dict:
            return loss_dict

        x = batch_dict["x"].to(self.device)
        gt = batch_dict["gt"].to(self.device)

        de_smoke_x, f, t = self.forward_dict(batch_dict)

        loss_con = F.mse_loss(x, de_smoke_x.mul(t).add(f))

        loss_rec = F.mse_loss(gt, de_smoke_x)

        loss_svp = self.get_svp_loss(f)

        loss_grad = loss_con + loss_rec + loss_svp
        loss_sum = loss_con + loss_rec + loss_svp

        loss_dict["Grad"] = loss_grad
        loss_dict["Loss"] = loss_sum
        loss_dict["Rec"] = loss_rec
        loss_dict["Con"] = loss_con
        loss_dict["SVP"] = loss_svp

        return loss_dict


class SVPNet_SurgiATM(SVPNet):
    def __init__(self, dc_window_size=15, dc_bias=0):
        super().__init__()

        self.surgi_atm = SurgiATM(dc_window_size=dc_window_size, dc_bias=dc_bias)

    def forward(self, x):
        # [B, 3, H, W]

        output, f, t = self.net(x)

        pre_gt, dcs, sdc = self.surgi_atm(x, output)

        return pre_gt, dcs, sdc, f, t

    def forward_dict(self, batch_dict):
        x = batch_dict["x"].to(self.device)
        # [B, 3, H, W]

        de_smoke_x, dcs, sdc, f, t = self.forward(x)
        batch_dict["pre_gt"] = de_smoke_x.detach()

        batch_dict["vis"] = {}
        batch_dict["vis"]["F"] = f.detach()
        batch_dict["vis"]["t"] = t.detach()
        batch_dict["vis"]["dc_scalar"] = dcs.detach()
        batch_dict["vis"]["scaled_dc"] = sdc.detach()

        return de_smoke_x, f, t