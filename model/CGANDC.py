"""
Following the work:

S. Salazar-Colores, H. M. Jiménez, C. J. Ortiz-Echeverri and G. Flores, "Desmoking Laparoscopy Surgery Images Using an
Image-to-Image Translation Guided by an Embedded Dark Channel," in IEEE Access, vol. 8, pp. 208898-208909,
2020, doi: 10.1109/ACCESS.2020.3038437.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from model.BasedModel import BaseModel, ChannelLayerNorm
import timm
import torchvision.transforms.functional as TF

from model.SurgiATM import SurgiATM
from utils.tool_functions import get_dcp_tensor


def build_gd_encoder_block(in_dim, out_dim):
    block = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, 4, 2, padding=1),
        nn.BatchNorm2d(out_dim),
        nn.LeakyReLU(inplace=True),
    )
    return block


def build_gd_decoder_block(in_dim, out_dim, dropout=False):
    block = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_dim),
        nn.LeakyReLU(inplace=True),
        nn.Dropout2d(0.5) if dropout else nn.Identity(),
    )
    return block


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.en_block1 = build_gd_encoder_block(4, 64)
        self.en_block2 = build_gd_encoder_block(64, 128)
        self.en_block3 = build_gd_encoder_block(128, 256)
        self.en_block4 = build_gd_encoder_block(256, 512)
        self.en_block5 = build_gd_encoder_block(512, 512)
        self.en_block6 = build_gd_encoder_block(512, 512)
        self.en_block7 = build_gd_encoder_block(512, 512)

        self.bottle_neck = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 1024, 1, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(0.5)
        )

        self.de_block7 = build_gd_decoder_block(1024, 1024, dropout=True)
        self.de_block6 = build_gd_decoder_block(1024 + 512, 1024, dropout=True)
        self.de_block5 = build_gd_decoder_block(1024 + 512, 1024)
        self.de_block4 = build_gd_decoder_block(1024 + 512, 1024)
        self.de_block3 = build_gd_decoder_block(1024 + 512, 512)
        self.de_block2 = build_gd_decoder_block(512 + 256, 256)
        self.de_block1 = build_gd_decoder_block(256 + 128, 128)

        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(128 + 64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # [B, 4, H, W]

        mark = False
        if len(x) == 1 and self.training:
            mark = True
            x = torch.cat([x, x], dim=0)

        en1 = self.en_block1(x)
        en2 = self.en_block2(en1)
        en3 = self.en_block3(en2)
        en4 = self.en_block4(en3)
        en5 = self.en_block5(en4)
        en6 = self.en_block6(en5)
        en7 = self.en_block7(en6)

        neck = self.bottle_neck(en7)

        de7 = self.de_block7(neck)
        if de7.size()[2:] != en7.size()[2:]:
            de7 = F.interpolate(de7, en7.size()[2:], mode="bilinear")
        de7 = torch.cat([de7, en7], dim=1)

        de6 = self.de_block6(de7)
        if de6.size()[2:] != en6.size()[2:]:
            de6 = F.interpolate(de6, en6.size()[2:], mode="bilinear")
        de6 = torch.cat([de6, en6], dim=1)

        de5 = self.de_block5(de6)
        if de5.size()[2:] != en5.size()[2:]:
            de5 = F.interpolate(de5, en5.size()[2:], mode="bilinear")
        de5 = torch.cat([de5, en5], dim=1)

        de4 = self.de_block4(de5)
        if de4.size()[2:] != en4.size()[2:]:
            de4 = F.interpolate(de4, en4.size()[2:], mode="bilinear")
        de4 = torch.cat([de4, en4], dim=1)

        de3 = self.de_block3(de4)
        if de3.size()[2:] != en3.size()[2:]:
            de3 = F.interpolate(de3, en3.size()[2:], mode="bilinear")
        de3 = torch.cat([de3, en3], dim=1)

        de2 = self.de_block2(de3)
        if de2.size()[2:] != en2.size()[2:]:
            de2 = F.interpolate(de2, en2.size()[2:], mode="bilinear")
        de2 = torch.cat([de2, en2], dim=1)

        de1 = self.de_block1(de2)
        if de1.size()[2:] != en1.size()[2:]:
            de1 = F.interpolate(de1, en1.size()[2:], mode="bilinear")
        de1 = torch.cat([de1, en1], dim=1)

        output = self.output_layer(de1)

        if output.size()[2:] != x.size()[2:]:
            output = F.interpolate(output, x.size()[2:], mode="bilinear")

        if mark:
            output = output[0:1]

        return output


def build_d_encoder_block(in_dim, out_dim, bn=True):
    block = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, 4, 2, padding=1),
        nn.BatchNorm2d(out_dim) if bn else nn.Identity(),     # use BN in D is not a good idea
        nn.LeakyReLU(inplace=True),
    )
    return block


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.D = nn.Sequential(
            build_d_encoder_block(6, 64, bn=True),
            build_d_encoder_block(64, 128, bn=True),
            build_d_encoder_block(128, 256, bn=True),
            nn.Conv2d(256, 512, 4, 1, 2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
        )

        self.output_layer = nn.Sequential(
            nn.Conv2d(512, 1, 1, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # [B, 3, H, W]

        feat = self.D(x)

        output = self.output_layer(feat)

        return output


class CGANDC(BaseModel):
    def __init__(self):
        super().__init__()

        self.G = Generator()
        self.D = Discriminator()

        self.frozen_module_list = nn.ModuleList([])
        self.trained_module_list = nn.ModuleList([self.G, self.D])
        self.param_module_list = nn.ModuleList([self.G])

        self.device = "cpu"

    def get_dcp_r(self, x, s=15):
        kernel_size = s

        b, c, h, w = x.shape

        dcp = get_dcp_tensor(x, s=kernel_size)

        x = TF.rgb_to_grayscale(x)

        x_dcp = x * dcp

        pad_s = (kernel_size - 1) // 2

        x_pad = F.pad(x, (pad_s, pad_s, pad_s, pad_s), mode="reflect")
        dcp_pad = F.pad(dcp, (pad_s, pad_s, pad_s, pad_s), mode="reflect")
        x_dcp_pad = F.pad(x_dcp, (pad_s, pad_s, pad_s, pad_s), mode="reflect")

        x_patches = F.unfold(x_pad, kernel_size=kernel_size, stride=1, padding=0)

        patch_n = x_patches.shape[-1]

        x_patches = x_patches.reshape(b, 1, kernel_size ** 2, patch_n)

        x_dcp_avg = F.avg_pool2d(x_dcp_pad, kernel_size=kernel_size, stride=1, padding=0)
        x_avg = F.avg_pool2d(x_pad, kernel_size=kernel_size, stride=1, padding=0)
        dcp_avg = F.avg_pool2d(dcp_pad, kernel_size=kernel_size, stride=1, padding=0)
        x_var = x_patches.var(dim=-2) + 1e-5
        x_var = x_var.reshape(b, 1, h, w)

        a = (x_dcp_avg - x_avg * dcp_avg) / x_var
        b = dcp_avg - a * x_avg

        dcp_r = a * x + b

        return dcp_r

    def forward(self, x):
        # [B, 3, H, W]

        dcp_r = self.get_dcp_r(x)

        x_in = torch.cat([x, dcp_r], dim=1)

        de_smoke_x = self.G(x_in)

        return de_smoke_x

    def forward_d(self, c, x):
        x_in = torch.cat([c, x], dim=1)

        output = self.D(x_in)

        return output

    def get_loss_d(self, d_x_gt, d_de_smoke_x):
        ones_tensor = torch.ones_like(d_x_gt, device=self.device)
        zeros_tensor = torch.zeros_like(d_x_gt, device=self.device)

        l_d = F.binary_cross_entropy(d_x_gt, ones_tensor) + F.binary_cross_entropy(d_de_smoke_x, zeros_tensor)

        return l_d

    def get_loss_g(self, d_de_smoke_x):
        ones_tensor = torch.ones_like(d_de_smoke_x, device=self.device)

        l_g = F.binary_cross_entropy(d_de_smoke_x, ones_tensor)

        return l_g

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
            "L1": 0,
        }

        if only_dict:
            return loss_dict

        x = batch_dict["x"].to(self.device)
        gt = batch_dict["gt"].to(self.device)

        de_smoke_x = self.forward_dict(batch_dict)
        d_x_gt = self.forward_d(gt, x)

        d_de_smoke_x = self.forward_d(de_smoke_x, x)
        loss_g = self.get_loss_g(d_de_smoke_x)

        # d_de_smoke_x = self.forward_d(de_smoke_x.detach(), x)
        loss_d = self.get_loss_d(d_x_gt, d_de_smoke_x)

        loss_l1 = F.l1_loss(de_smoke_x, gt)

        loss_grad = loss_d + loss_g + loss_l1
        loss_sum = loss_d + loss_g + loss_l1

        loss_dict["Grad"] = loss_grad
        loss_dict["Loss"] = loss_sum
        loss_dict["D"] = loss_d
        loss_dict["G"] = loss_g
        loss_dict["L1"] = loss_l1

        return loss_dict


class CGANDC_SurgiATM(CGANDC):
    def __init__(self, dc_window_size=15, dc_bias=0):
        super().__init__()

        self.surgi_atm = SurgiATM(dc_window_size=dc_window_size, dc_bias=dc_bias)

    def forward(self, x):
        # [B, 3, H, W]
        dcp_r = self.get_dcp_r(x)

        x_in = torch.cat([x, dcp_r], dim=1)

        output = self.G(x_in)

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
