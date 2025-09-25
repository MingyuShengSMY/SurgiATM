"""
Following the work:

Wang, W., Liu, F., Hao, J., Yu, X., Zhang, B. and Shi, C., 2024.
Desmoking of the Endoscopic Surgery Images Based on A Local-Global U-Shaped Transformer Model.
IEEE Transactions on Medical Robotics and Bionics.
"""
from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from torchvision.models.swin_transformer import SwinTransformerBlock, SwinTransformer
import torchvision.models as TVM
from timm.models.swin_transformer import WindowAttention
from model.BasedModel import BaseModel
import timm
import torchvision.transforms.functional as TF

from model.SurgiATM import SurgiATM


class nnPermute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class SKFusion(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim

        self.f_layer = nn.Conv2d(in_dim, in_dim, 1, 1)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_dim, in_dim * 2, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_dim * 2, in_dim * 2, 1, 1),
        )

    def forward(self, x1, x2):
        # [B, C, H, W]

        b, c, h, w = x1.shape

        fx1 = self.f_layer(x1)

        if fx1.size()[2:] != x2.size()[2:]:
            fx1 = F.interpolate(fx1, x2.size()[2:], mode="bilinear")

        x12 = fx1 + x2

        x_gap = self.gap(x12)

        a12: torch.Tensor = self.mlp(x_gap).reshape(b, c, 2, 1, 1)  # [B, C, 2, 1, 1]

        a12 = a12.softmax(dim=2)

        a1, a2 = a12.unbind(dim=2)  # [B, C, 1, 1]

        y = fx1 * a1 + x2 * a2 + x2

        return y


class LeFF(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.in_dim = in_dim

        self.fc1 = nn.Conv2d(in_dim, in_dim, 1, 1)

        self.d_conv3 = nn.Conv2d(in_dim, in_dim, 3, 1, padding=1, groups=in_dim)
        self.d_conv5 = nn.Conv2d(in_dim, in_dim, 5, 1, padding=2, groups=in_dim)

        self.fc2 = nn.Conv2d(in_dim * 2, in_dim, 1, 1)

    def forward(self, x):
        # [B, C, H, W]

        fc1 = self.fc1(x)

        dc3 = self.d_conv3(fc1)
        dc5 = self.d_conv5(fc1)

        dc = torch.cat([dc3, dc5], dim=1)

        fc2 = self.fc2(dc)

        return fc2


class LGTBlock(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim

        self.per_ln1 = nn.Sequential(
            nnPermute(0, 2, 3, 1),
            nn.LayerNorm(in_dim),
        )

        self.w_msa1 = nn.Sequential(
            SwinTransformerBlock(in_dim, 6, [8, 8], [0, 0]),
        )
        self.w_msa2 = nn.Sequential(
            nnPermute(0, 3, 1, 2),
            nn.UpsamplingBilinear2d(scale_factor=0.5),
            nnPermute(0, 2, 3, 1),
            SwinTransformerBlock(in_dim, 6, [8, 8], [0, 0]),
            nnPermute(0, 3, 1, 2),
            nn.UpsamplingBilinear2d(scale_factor=2.0),
            nnPermute(0, 2, 3, 1),
        )

        self.ln_leff1 = nn.Sequential(
            nn.LayerNorm(in_dim),
            nnPermute(0, 3, 1, 2),
            LeFF(in_dim),
        )

        self.per_ln2 = nn.Sequential(
            nnPermute(0, 2, 3, 1),
            nn.LayerNorm(in_dim),
        )

        self.sw_msa1 = nn.Sequential(
            SwinTransformerBlock(in_dim, 6, [8, 8], [4, 4]),
        )
        self.sw_msa2 = nn.Sequential(
            nnPermute(0, 3, 1, 2),
            nn.UpsamplingBilinear2d(scale_factor=0.5),
            nnPermute(0, 2, 3, 1),
            SwinTransformerBlock(in_dim, 6, [8, 8], [4, 4]),
            nnPermute(0, 3, 1, 2),
            nn.UpsamplingBilinear2d(scale_factor=2.0),
            nnPermute(0, 2, 3, 1),
        )

        self.ln_leff2 = nn.Sequential(
            nn.LayerNorm(in_dim),
            nnPermute(0, 3, 1, 2),
            LeFF(in_dim),
        )

    def forward(self, x):
        # [B, C, H, W]

        x_ln1 = self.per_ln1(x)

        wm1 = self.w_msa1(x_ln1)
        wm2 = self.w_msa2(x_ln1)

        if wm2.permute(0, 3, 1, 2).size()[2:] != x.size()[2:]:
            wm2 = F.interpolate(wm2.permute(0, 3, 1, 2), x.size()[2:], mode="bilinear").permute(0, 2, 3, 1)

        if wm1.permute(0, 3, 1, 2).size()[2:] != x.size()[2:]:
            wm1 = F.interpolate(wm1.permute(0, 3, 1, 2), x.size()[2:], mode="bilinear").permute(0, 2, 3, 1)

        xwm = x.permute(0, 2, 3, 1) + wm1 + wm2

        lef = self.ln_leff1(xwm)

        z1 = lef + xwm.permute(0, 3, 1, 2)

        x_ln2 = self.per_ln2(z1)

        swm1 = self.sw_msa1(x_ln2)
        swm2 = self.sw_msa2(x_ln2)

        if swm2.permute(0, 3, 1, 2).size()[2:] != z1.size()[2:]:
            swm2 = F.interpolate(swm2.permute(0, 3, 1, 2), z1.size()[2:], mode="bilinear").permute(0, 2, 3, 1)

        if swm2.permute(0, 3, 1, 2).size()[2:] != z1.size()[2:]:
            swm2 = F.interpolate(swm2.permute(0, 3, 1, 2), z1.size()[2:], mode="bilinear").permute(0, 2, 3, 1)

        xswm = z1.permute(0, 2, 3, 1) + swm1 + swm2

        lef = self.ln_leff2(xswm)

        if lef.size()[2:] != xswm.permute(0, 3, 1, 2).size()[2:]:
            lef = F.interpolate(lef, xswm.permute(0, 3, 1, 2).size()[2:], mode="bilinear")

        z2 = lef + xswm.permute(0, 3, 1, 2)

        return z2


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        base_dim = 42

        self.base_dim = base_dim

        self.input_proj = nn.Sequential(
            nn.Conv2d(3, base_dim, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
        )

        self.en_lg_block1 = nn.Sequential(
            LGTBlock(base_dim),
            LGTBlock(base_dim),
        )

        self.down1 = nn.Conv2d(base_dim, base_dim * 2, 4, 2, 1)

        self.en_lg_block2 = nn.Sequential(
            LGTBlock(base_dim * 2),
            LGTBlock(base_dim * 2),
        )

        self.down2 = nn.Conv2d(base_dim * 2, base_dim * 4, 4, 2, 1)

        self.neck_lg_block = nn.Sequential(
            LGTBlock(base_dim * 4),
            LGTBlock(base_dim * 4),
        )

        self.up2 = nn.ConvTranspose2d(base_dim * 4, base_dim * 2, 2, 2)

        self.skf2 = SKFusion(base_dim * 2)

        self.de_lg_block2 = nn.Sequential(
            LGTBlock(base_dim * 2),
            LGTBlock(base_dim * 2),
        )

        self.up1 = nn.ConvTranspose2d(base_dim * 2, base_dim, 2, 2)

        self.skf1 = SKFusion(base_dim)

        self.de_lg_block1 = nn.Sequential(
            LGTBlock(base_dim),
            LGTBlock(base_dim),
        )

        self.output_proj = nn.Sequential(
            nn.Conv2d(base_dim, 3, 3, 1, 1),
        )

    def forward(self, x):
        # [B, 3, H, W]

        x_in = self.input_proj(x)

        en_lg1 = self.en_lg_block1(x_in)

        en_lg2 = self.en_lg_block2(self.down1(en_lg1))

        neck_lg = self.up2(self.neck_lg_block(self.down2(en_lg2)))

        sk2 = self.skf2(en_lg2, neck_lg)

        de_lg2 = self.up1(self.de_lg_block2(sk2))

        sk1 = self.skf1(en_lg1, de_lg2)

        de_lg1 = self.de_lg_block1(sk1)

        output = self.output_proj(de_lg1)

        if output.size()[2:] != x.size()[2:]:
            output = F.interpolate(output, x.size()[2:], mode="bilinear")

        return output


class LGUTransformer(BaseModel):
    def __init__(self):
        super().__init__()

        self.net = Net()

        self.frozen_module_list = nn.ModuleList([])
        self.trained_module_list = nn.ModuleList([self.net])
        self.param_module_list = nn.ModuleList([self.net])

        self.device = "cpu"

    def forward(self, x):
        # [B, 3, H, W]

        res = self.net(x)

        if res.size()[2:] != x.size()[2:]:
            res = F.interpolate(res, x.size()[2:], mode="bilinear")

        pre_gt = x + res

        return pre_gt, res

    def forward_dict(self, batch_dict):
        x = batch_dict["x"].to(self.device)
        # [B, 3, H, W]

        pre_gt, res = self.forward(x)
        batch_dict["pre_gt"] = pre_gt.detach()
        batch_dict["vis"] = {}

        batch_dict["vis"]["res"] = res.detach()
        return pre_gt

    def get_loss_dict(self, batch_dict, only_dict=False):
        loss_dict = {
            "Grad": 0,
            "Loss": 0,
        }

        if only_dict:
            return loss_dict

        x = batch_dict["x"].to(self.device)
        gt = batch_dict["gt"].to(self.device)

        pre_gt = self.forward_dict(batch_dict)

        loss_char = pre_gt.sub(gt).square().add(1e-6).sqrt().mean()

        loss_grad = loss_char
        loss_sum = loss_char

        loss_dict["Grad"] = loss_grad
        loss_dict["Loss"] = loss_sum

        return loss_dict


class LGUTransformer_SurgiATM(LGUTransformer):
    def __init__(self, dc_window_size=15, dc_bias=0):
        super().__init__()

        self.surgi_atm = SurgiATM(dc_window_size=dc_window_size, dc_bias=dc_bias)

    def forward(self, x):
        # [B, 3, H, W]

        output = self.net(x)

        output = output.sigmoid()

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

