"""
Following the work:

Sidorov, O., Wang, C. and Cheikh, F.A., 2020, April. Generative smoke removal.
In Machine Learning for Health Workshop (pp. 81-92). PMLR.

Derived from their GitHub:
https://github.com/acecreamu/ssim-pan
"""
from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from model.BasedModel import BaseModel
import timm
import torchvision.transforms.functional as TF

from model.SurgiATM import SurgiATM
from utils.tool_functions import get_dcp_tensor, ms_ssim_bchw_tensor_01


def build_gd_encoder_block(in_dim, out_dim, stride=2):
    block = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, 4, stride=stride, padding=1),
        nn.BatchNorm2d(out_dim),
        nn.LeakyReLU(inplace=True),
    )
    return block


def build_gd_decoder_block(in_dim, out_dim, dropout=False, act_mark=True, bn=True):
    block = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_dim) if bn else nn.Identity(),
        nn.LeakyReLU(inplace=True) if act_mark else nn.Identity(),
    )
    return block


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.en_block1 = build_gd_encoder_block(3, 64)
        self.en_block2 = build_gd_encoder_block(64, 128)
        self.en_block3 = build_gd_encoder_block(128, 256)
        self.en_block4 = build_gd_encoder_block(256, 512)
        self.en_block5 = build_gd_encoder_block(512, 512)
        self.en_block6 = build_gd_encoder_block(512, 512)

        self.de_block6 = build_gd_decoder_block(512, 512)
        self.de_block5 = build_gd_decoder_block(512 + 512, 512)
        self.de_block4 = build_gd_decoder_block(512 + 512, 256)
        self.de_block3 = build_gd_decoder_block(256 + 256, 128)
        self.de_block2 = build_gd_decoder_block(128 + 128, 64)
        self.de_block1 = build_gd_decoder_block(64, 3, act_mark=False, bn=False)

        self.output_layer = nn.Sequential(
            nn.Sigmoid(),
        )

    def forward(self, x):
        # [B, 4, H, W]

        en1 = self.en_block1(x)
        en2 = self.en_block2(en1)
        en3 = self.en_block3(en2)
        en4 = self.en_block4(en3)
        en5 = self.en_block5(en4)
        en6 = self.en_block6(en5)

        de6 = self.de_block6(en6)
        if de6.size()[2:] != en5.size()[2:]:
            de6 = TF.resize(de6, en5.size()[2:])
        de6 = torch.cat([de6, en5], dim=1)

        de5 = self.de_block5(de6)
        if de5.size()[2:] != en4.size()[2:]:
            de5 = TF.resize(de5, en4.size()[2:])
        de5 = torch.cat([de5, en4], dim=1)

        de4 = self.de_block4(de5)
        if de4.size()[2:] != en3.size()[2:]:
            de4 = TF.resize(de4, en3.size()[2:])
        de4 = torch.cat([de4, en3], dim=1)

        de3 = self.de_block3(de4)
        if de3.size()[2:] != en2.size()[2:]:
            de3 = TF.resize(de3, en2.size()[2:])
        de3 = torch.cat([de3, en2], dim=1)

        de2 = self.de_block2(de3)
        de1 = self.de_block1(de2)

        output = self.output_layer(de1)

        if output.size()[2:] != x.size()[2:]:
            output = TF.resize(output, x.size()[2:])

        return output


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),  # 128
            nn.LeakyReLU(0.2, True),
        )

        self.block1 = nn.Sequential(
            build_gd_encoder_block(64, 128),  # 64
            build_gd_encoder_block(128, 128, stride=1),
            build_gd_encoder_block(128, 256),  # 32
        )
        self.block2 = nn.Sequential(
            build_gd_encoder_block(256, 256, stride=1),
            build_gd_encoder_block(256, 512),  # 16
        )
        self.block3 = nn.Sequential(
            build_gd_encoder_block(512, 512, stride=1),
            build_gd_encoder_block(512, 512),  # 8
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(512, 8, 3, 2, 1),  # 4
            nn.BatchNorm2d(8),
            nn.LeakyReLU(inplace=True)
        )

        self.output_layer = nn.Sequential(
            nn.Linear(4 * 4 * 8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # [B, 3, H, W]

        per1 = self.input_layer(x)

        per2 = self.block1(per1)
        per3 = self.block2(per2)
        per4 = self.block3(per3)
        feat = self.block4(per4)

        feat_flatten = feat.flatten(1, -1)

        output = self.output_layer(feat_flatten)

        per_list = [per1, per2, per3, per4]

        return output, per_list


class SSIM_PAN(BaseModel):
    def __init__(self):
        super().__init__()

        self.G = Generator()
        self.D = Discriminator()

        self.frozen_module_list = nn.ModuleList([])
        self.trained_module_list = nn.ModuleList([self.G, self.D])
        self.param_module_list = nn.ModuleList([self.G])

        self.device = "cpu"

    def config_opt(self, opt_config):
        opt_list = [
            torch.optim.Adam(self.G.parameters(), lr=opt_config["lr"], betas=(opt_config["beta1"], opt_config["beta2"])),
            torch.optim.Adam(self.D.parameters(), lr=opt_config["lr"], betas=(opt_config["beta1"], opt_config["beta2"])),
        ]
        return opt_list

    def forward(self, x):  # default is smoke to smoke_less, s2sl
        # [B, 3, H, W]

        de_smoke_x = self.G(x)

        return de_smoke_x

    def forward_d(self, x):
        # [B, 3, H, W]

        output, per_list = self.D(x)

        return output, per_list

    def get_loss_d(self, d_x_gt, d_de_smoke_x):
        ones_tensor = torch.ones_like(d_x_gt, device=self.device)
        zeros_tensor = torch.zeros_like(d_x_gt, device=self.device)

        l_d = F.binary_cross_entropy(d_x_gt, ones_tensor) + F.binary_cross_entropy(d_de_smoke_x, zeros_tensor)
        return l_d

    def get_loss_g(self, d_de_smoke_x):
        ones_tensor = torch.ones_like(d_de_smoke_x, device=self.device)
        l_g = F.binary_cross_entropy(d_de_smoke_x, ones_tensor)
        return l_g

    def __gaussian__(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def __create_window__(self, window_size, channel=1):
        _1D_window = self.__gaussian__(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def __ssim__(self, img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
        # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
        if val_range is None:
            if torch.max(img1) > 128:
                max_val = 255
            else:
                max_val = 1

            if torch.min(img1) < -0.5:
                min_val = -1
            else:
                min_val = 0
            L = max_val - min_val
        else:
            L = val_range

        padd = 0
        (_, channel, height, width) = img1.size()
        if window is None:
            real_size = min(window_size, height, width)
            window = self.__create_window__(real_size, channel=channel).to(img1.device)

        mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
        mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)

        if full:
            return ret, cs
        return ret

    def __msssim__(self, img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
        device = img1.device
        weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
        levels = weights.size()[0]
        mssim = []
        mcs = []
        for _ in range(levels):
            sim, cs = self.__ssim__(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
            mssim.append(sim)
            mcs.append(cs)

            img1 = F.avg_pool2d(img1, (2, 2))
            img2 = F.avg_pool2d(img2, (2, 2))

        mssim = torch.stack(mssim)
        mcs = torch.stack(mcs)

        # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
        if normalize:
            mssim = (mssim + 1) / 2
            mcs = (mcs + 1) / 2

        pow1 = mcs ** weights
        pow2 = mssim ** weights
        # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
        output = torch.prod(pow1[:-1] * pow2[-1])

        return output

    def get_loss_ssim(self, de_smoke_x, gt):
        loss_ssim = 1 - self.__msssim__(de_smoke_x, gt, normalize=True)
        return loss_ssim

    def get_loss_pan(self, de_smoke_x_per_list, gt_per_list):
        loss_pan = 0.0

        for per1, per2 in zip(de_smoke_x_per_list, gt_per_list):
            loss_pan += F.l1_loss(per1, per2.detach())

        return loss_pan

    def forward_dict(self, batch_dict):
        x = batch_dict["x"].to(self.device)
        # [B, 3, H, W]

        de_smoke_x = self.forward(x)
        batch_dict["pre_gt"] = de_smoke_x.detach()

        return de_smoke_x

    def get_loss_dict(self, batch_dict, only_dict=False):
        loss_dict = {
            "Grad": 0,
            "Loss": 0,
            "D": 0,
            "G": 0,
            "MS-SSIM": 0,
            "PAN": 0,
            "L1": 0,
        }

        if only_dict:
            return loss_dict

        x = batch_dict["x"].to(self.device)
        gt = batch_dict["gt"].to(self.device)

        de_smoke_x = self.forward_dict(batch_dict)

        loss_l1 = F.l1_loss(gt, de_smoke_x)

        d_x_gt, gt_per_list = self.forward_d(gt)
        d_de_smoke_x, de_smoke_x_per_list = self.forward_d(de_smoke_x)

        loss_d = self.get_loss_d(d_x_gt, d_de_smoke_x)

        loss_g = self.get_loss_g(d_de_smoke_x)

        loss_ssim = self.get_loss_ssim(de_smoke_x, gt)

        loss_pan = self.get_loss_pan(de_smoke_x_per_list, gt_per_list)

        loss_grad = loss_d + loss_g + loss_ssim + loss_pan + loss_l1
        loss_sum = loss_d + loss_g + loss_ssim + loss_pan + loss_l1

        loss_dict["Grad"] = loss_grad
        loss_dict["Loss"] = loss_sum
        loss_dict["D"] = loss_d
        loss_dict["G"] = loss_g
        loss_dict["MS-SSIM"] = loss_ssim
        loss_dict["PAN"] = loss_pan
        loss_dict["L1"] = loss_l1

        return loss_dict


class SSIM_PAN_SurgiATM(SSIM_PAN):
    def __init__(self, dc_window_size=15, dc_bias=0):
        super().__init__()

        self.surgi_atm = SurgiATM(dc_window_size=dc_window_size, dc_bias=dc_bias)

    def forward(self, x):
        # [B, 3, H, W]

        output = self.G(x)

        pre_gt, dcs, sdc = self.surgi_atm(x, output)

        return pre_gt, dcs, sdc

    def forward_dict(self, batch_dict):
        x = batch_dict["x"].to(self.device)
        # [B, 3, H, W]

        pre_gt, dcs, sdc = self.forward(x)
        batch_dict["pre_gt"] = pre_gt.detach()

        batch_dict["vis"] = {}

        batch_dict["vis"]["dc_scalar"] = dcs.detach()
        batch_dict["vis"]["scaled_dc"] = sdc.detach()

        return pre_gt

