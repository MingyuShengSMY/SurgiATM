"""
Following the work De-smokeGCN:
Chen, L., Tang, W., John, N.W., Wan, T.R. and Zhang, J.J., 2019. De-smokeGCN: generative cooperative networks for
joint surgical smoke detection and removal. IEEE transactions on medical imaging, 39(5), pp.1615-1625.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.BasedModel import BaseModel
from model.SurgiATM import SurgiATM
from utils.tool_functions import get_dcp_tensor


def build_encoder_block(in_dim, out_dim, k_size, padding, bn=True, act_fun=True):
    block = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=k_size, stride=2, padding=padding, bias=False),
        nn.BatchNorm2d(out_dim) if bn else nn.Identity(),
        nn.LeakyReLU(inplace=True) if act_fun else nn.Identity(),
    )
    return block


def build_decoder_block(in_dim, out_dim, k_size, padding, bn=True, act_fun=True):
    block = nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=k_size, stride=2, padding=padding, bias=False),
        nn.BatchNorm2d(out_dim) if bn else nn.Identity(),
        nn.LeakyReLU(inplace=True) if act_fun else nn.Identity(),
    )
    return block


class Detector(nn.Module):
    def __init__(self):
        super().__init__()

        self.en_block1 = build_encoder_block(3, 64, 4, 1, bn=True)  # H / 2
        self.en_block2 = build_encoder_block(64, 128, 4, 1, bn=True)  # H / 4
        self.en_block3 = build_encoder_block(128, 256, 4, 1, bn=True)  # H / 8
        self.en_block4 = build_encoder_block(256, 512, 4, 1, bn=True)  # H / 16

        self.drop_out = nn.Dropout2d(p=0.5)

        self.de_block4 = build_decoder_block(512, 256, 4, 1, bn=True)  # H / 8
        self.de_block3 = build_decoder_block(256 * 2, 128, 4, 1, bn=True)  # H / 4
        self.de_block2 = build_decoder_block(128 * 2, 64, 4, 1, bn=True)  # H / 2
        self.de_block1 = build_decoder_block(64 * 2, 1, 4, 1, act_fun=False, bn=False)  # H / 16

        self.output_layer = nn.Sequential(
            nn.Sigmoid(),
        )

    def forward(self, x):
        # [B, 3, H, W]

        en1 = self.en_block1(x)
        en2 = self.en_block2(en1)
        en3 = self.en_block3(en2)
        dense_feature = self.en_block4(en3)

        dense_feature = self.drop_out(dense_feature)

        de3 = self.de_block4(dense_feature)
        if de3.size()[2:] != en3.size()[2:]:
            de3 = F.interpolate(de3, en3.size()[2:], mode="bilinear")
        de2 = self.de_block3(torch.cat([de3, en3], dim=1))
        if de2.size()[2:] != en2.size()[2:]:
            de2 = F.interpolate(de2, en2.size()[2:], mode="bilinear")
        de1 = self.de_block2(torch.cat([de2, en2], dim=1))
        if de1.size()[2:] != en1.size()[2:]:
            de1 = F.interpolate(de1, en1.size()[2:], mode="bilinear")
        de0 = self.de_block1(torch.cat([de1, en1], dim=1))

        output = self.output_layer(de0)

        if output.size()[2:] != x.size()[2:]:
            output = F.interpolate(output, x.size()[2:], mode="bilinear")

        return output

    def get_loss(self, y_pre, y_gt):
        # [B, 3, H, W]
        # [B, 1, H, W]

        l1_loss = F.l1_loss(y_pre, y_gt)
        smooth_x = F.l1_loss(y_pre[:, :, 1:, :], y_pre[:, :, :-1, :])
        smooth_y = F.l1_loss(y_pre[:, :, :, 1:], y_pre[:, :, :, :-1])

        loss = l1_loss + smooth_x + smooth_y

        return loss


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.en_block1 = build_encoder_block(4, 64, 4, 1, bn=True)  # H / 2
        self.en_block2 = build_encoder_block(64, 128, 4, 1, bn=True)  # H / 4
        self.en_block3 = build_encoder_block(128, 256, 4, 1, bn=True)  # H / 8
        self.en_block4 = build_encoder_block(256, 512, 4, 1, bn=True)  # H / 16
        self.en_block5 = build_encoder_block(512, 512, 4, 1, bn=True)  # H / 32
        self.en_block6 = build_encoder_block(512, 512, 4, 1, bn=True)  # H / 64
        self.en_block7 = build_encoder_block(512, 512, 4, 1, bn=True)  # H / 128
        self.en_block8 = build_encoder_block(512, 512, 4, 1, bn=True)  # H / 256

        self.drop_out = nn.Dropout2d(p=0.5)

        self.de_block8 = build_decoder_block(512, 512, 4, 1, bn=True)  # H / 128
        self.de_block7 = build_decoder_block(512 * 2, 512, 4, 1, bn=True)  # H / 64
        self.de_block6 = build_decoder_block(512 * 2, 512, 4, 1, bn=True)  # H / 32
        self.de_block5 = build_decoder_block(512 * 2, 512, 4, 1, bn=True)  # H / 16
        self.de_block4 = build_decoder_block(512 * 2, 256, 4, 1, bn=True)  # H / 8
        self.de_block3 = build_decoder_block(256 * 2, 128, 4, 1, bn=True)  # H / 4
        self.de_block2 = build_decoder_block(128 * 2, 64, 4, 1, bn=True)  # H / 2
        self.de_block1 = build_decoder_block(64 * 2, 3, 4, 1, act_fun=False, bn=False)  # H

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
        en7 = self.en_block7(en6)

        if min(en7.size()[2:]) < 2:
            scale_factor = 4 / min(en7.size()[2:])
            en7 = F.interpolate(en7, scale_factor=(scale_factor, scale_factor), mode="bilinear")

        dense_feature = self.en_block8(en7)

        dense_feature = self.drop_out(dense_feature)

        de7 = self.de_block8(dense_feature)
        if de7.size()[2:] != en7.size()[2:]:
            de7 = F.interpolate(de7, en7.size()[2:], mode="bilinear")
        de6 = self.de_block7(torch.cat([de7, en7], dim=1))
        if de6.size()[2:] != en6.size()[2:]:
            de6 = F.interpolate(de6, en6.size()[2:], mode="bilinear")
        de5 = self.de_block6(torch.cat([de6, en6], dim=1))
        if de5.size()[2:] != en5.size()[2:]:
            de5 = F.interpolate(de5, en5.size()[2:], mode="bilinear")
        de4 = self.de_block5(torch.cat([de5, en5], dim=1))
        if de4.size()[2:] != en4.size()[2:]:
            de4 = F.interpolate(de4, en4.size()[2:], mode="bilinear")
        de3 = self.de_block4(torch.cat([de4, en4], dim=1))
        if de3.size()[2:] != en3.size()[2:]:
            de3 = F.interpolate(de3, en3.size()[2:], mode="bilinear")
        de2 = self.de_block3(torch.cat([de3, en3], dim=1))
        if de2.size()[2:] != en2.size()[2:]:
            de2 = F.interpolate(de2, en2.size()[2:], mode="bilinear")
        de1 = self.de_block2(torch.cat([de2, en2], dim=1))
        if de1.size()[2:] != en1.size()[2:]:
            de1 = F.interpolate(de1, en1.size()[2:], mode="bilinear")
        de0 = self.de_block1(torch.cat([de1, en1], dim=1))

        output = self.output_layer(de0)

        if output.size()[2:] != x.size()[2:]:
            output = F.interpolate(output, x.size()[2:], mode="bilinear")

        return output

    def get_loss(self, y_pre, y_gt):
        # [B, 4, H, W]
        # [B, 3, H, W]

        l1_loss = F.l1_loss(y_pre, y_gt)

        loss = l1_loss

        return loss


class DesmokeGCN(BaseModel):
    def __init__(self):
        super().__init__()

        self.detector = Detector()
        self.generator = Generator()

        self.frozen_module_list = nn.ModuleList([])
        self.trained_module_list = nn.ModuleList([self.detector, self.generator])
        self.param_module_list = nn.ModuleList([self.generator])

        self.device = "cpu"

    def forward(self, x):
        # [B, 3, H, W]

        smoke_pre = self.detector(x)

        x_smoke_cat = torch.cat([x, smoke_pre.detach()], dim=1)

        de_smoke_x = self.generator(x_smoke_cat)

        return de_smoke_x, smoke_pre

    def dag(self, de_smoke_x):
        if self.training:
            self.detector.requires_grad_(False)

        remained_smoke_pre = self.detector(de_smoke_x)

        if self.training:
            self.detector.requires_grad_(True)

        return remained_smoke_pre

    def get_loss(self, de_smoke_x, smoke_pre, smoke_mask, gt, remained_smoke_pre):
        # [B, 3, H, W]
        # [B, 3, H, W]
        # [B, 1, H, W]

        loss_d = self.detector.get_loss(smoke_pre, smoke_mask)
        loss_g = self.generator.get_loss(de_smoke_x, gt)
        loss_dag = remained_smoke_pre.abs().mean()

        loss = loss_d + loss_g + loss_dag

        return loss, loss_d, loss_g, loss_dag

    def forward_dict(self, batch_dict):
        x = batch_dict["x"].to(self.device)
        # [B, 3, H, W]

        de_smoke_x, smoke_pre = self.forward(x)

        batch_dict["pre_gt"] = de_smoke_x.detach()

        batch_dict["vis_res"] = {}
        batch_dict["vis_res"]["pre_smoke"] = smoke_pre.detach()

        return de_smoke_x, smoke_pre

    def get_loss_dict(self, batch_dict, only_dict=False):
        loss_dict = {
            # "Grad": 0,
            "Loss": 0,
            "D": 0,
            "G": 0,
            "DaG": 0,
        }

        if only_dict:
            return loss_dict

        gt = batch_dict["gt"].to(self.device)
        smoke_mask = batch_dict["smoke"].to(self.device)

        de_smoke_x, smoke_pre = self.forward_dict(batch_dict)
        remained_smoke_pre = self.dag(de_smoke_x)

        loss, loss_d, loss_g, loss_dag = self.get_loss(de_smoke_x, smoke_pre, smoke_mask, gt, remained_smoke_pre)

        loss_sum = loss_d + loss_g + loss_dag
        loss_grad = loss_d + loss_g + loss_dag * 100  # only used for gradient update

        loss_dict["Grad"] = loss_grad
        loss_dict["Loss"] = loss_sum
        loss_dict["D"] = loss_d
        loss_dict["G"] = loss_g
        loss_dict["DaG"] = loss_dag

        return loss_dict


class DesmokeGCN_SurgiATM(DesmokeGCN):
    def __init__(self, dc_window_size=15, dc_bias=0):
        super().__init__()

        self.surgi_atm = SurgiATM(dc_window_size=dc_window_size, dc_bias=dc_bias)

    def forward(self, x):
        # [B, 3, H, W]

        pre_smoke = self.detector(x)

        x_smoke_cat = torch.cat([x, pre_smoke.detach()], dim=1)

        output = self.generator(x_smoke_cat)

        pre_gt, dcs, sdc = self.surgi_atm(x, output)

        return pre_gt, dcs, sdc, pre_smoke

    def forward_dict(self, batch_dict):
        x = batch_dict["x"].to(self.device)
        # [B, 3, H, W]

        pre_gt, dcs, sdc, pre_smoke = self.forward(x)
        batch_dict["pre_gt"] = pre_gt.detach()

        batch_dict["vis"] = {}

        batch_dict["vis"]["dc_scalar"] = dcs.detach()
        batch_dict["vis"]["scaled_dc"] = sdc.detach()
        batch_dict["vis"]["pre_smoke"] = pre_smoke.detach()

        return pre_gt, pre_smoke

