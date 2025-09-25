"""
Following the work:
T. Hong et al., "MARS-GAN: Multilevel-Feature-Learning Attention-Aware Based Generative Adversarial Network for
Removing Surgical Smoke," in IEEE Transactions on Medical Imaging, vol. 42, no. 8, pp. 2299-2312, Aug. 2023, doi:
10.1109/TMI.2023.3245298.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from model.BasedModel import BaseModel
from model.SurgiATM import SurgiATM
from utils.tool_functions import get_dcp_tensor


def build_encoder_block(in_dim, out_dim):
    block = nn.Sequential(
        nn.MaxPool2d(2, 2),
        nn.Conv2d(in_dim, out_dim, 3, 1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_dim, out_dim, 3, 1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_dim, out_dim, 3, 1, padding=1),
        nn.ReLU(inplace=True),
    )
    return block


def build_decoder_block(in_dim, out_dim):
    block = nn.Sequential(
        nn.Conv2d(in_dim, in_dim, 3, 1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_dim, in_dim, 3, 1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_dim, out_dim, 3, 1, padding=1),
        nn.ReLU(inplace=True),
        nn.UpsamplingNearest2d(scale_factor=2.0),
    )
    return block


class SSN(nn.Module):
    def __init__(self):
        super().__init__()

        self.en_input_block = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.en_block1 = build_encoder_block(128, 256)

        self.en_block2 = build_encoder_block(256, 512)

        self.en_block3 = build_encoder_block(512, 512)

        self.en_block4 = nn.MaxPool2d(2, 2)

        self.neck_block = nn.Sequential(
            nn.Conv2d(512, 512, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1, 1),
            nn.ReLU(inplace=True),
        )

        self.de_block4 = nn.Conv2d(512, 512, 3, 1, padding=1)

        self.de_block3 = nn.UpsamplingNearest2d(scale_factor=2.0)

        self.de_block2 = build_decoder_block(512, 256)

        self.de_block1 = build_decoder_block(256, 128)

        self.en2_512_256 = nn.Conv2d(512, 256, 1, 1)

        self.de_output_block = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2.0),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2.0),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, 1, padding=1),
            # nn.ReLU(inplace=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # [B, 3, H, W]

        en_in = self.en_input_block(x)
        en1 = self.en_block1(en_in)
        en2 = self.en_block2(en1)
        en3 = self.en_block3(en2)
        en4 = self.en_block4(en3)

        neck = self.neck_block(en4)

        if neck.size()[2:] != en4.size()[2:]:
            neck = F.interpolate(neck, en4.size()[2:], mode="bilinear")
        de4 = self.de_block4(neck + en4)

        de3 = self.de_block3(de4)
        if de3.size()[2:] != en3.size()[2:]:
            de3 = F.interpolate(de3, en3.size()[2:], mode="bilinear")
        de3 = de3 + en3

        de3_1 = self.de_block2(de3)
        en2_1 = self.en2_512_256(en2)
        if de3_1.size()[2:] != en2_1.size()[2:]:
            de3_1 = F.interpolate(de3_1, en2_1.size()[2:], mode="bilinear")

        de2 = de3_1 + en2_1

        de1 = self.de_block1(de2)
        de_out = self.de_output_block(de1)

        if de_out.size()[2:] != x.size()[2:]:
            de_out = F.interpolate(de_out, x.size()[2:], mode="bilinear")

        return de_out

    def get_loss(self, y_pre, y_gt):
        # [B, 3, H, W]
        # [B, 1, H, W]
        # [B, 1, H, W]

        loss = F.l1_loss(y_pre, y_gt)

        return loss


def build_multi_conv_block(in_dim, out_dim):
    block = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, 3, 1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_dim, out_dim, 3, 1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_dim, out_dim, 3, 1, padding=1),
        nn.ReLU(inplace=True),
    )
    return block


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, padding=1),
        )

    def forward(self, x):
        return self.block(x) + x


def build_residual_block(dim):
    return ResidualBlock(dim)


class MSF(nn.Module):
    def __init__(self):
        super().__init__()

        self.f1_block1 = nn.Sequential(
            build_multi_conv_block(5, 8),
            build_multi_conv_block(8, 8),
        )

        self.f1_block2 = nn.Sequential(
            build_residual_block(8),
            build_residual_block(8),
        )

        self.f2_block = nn.Sequential(
            build_multi_conv_block(5, 8),
            build_multi_conv_block(8, 8),
            build_residual_block(8),
            build_residual_block(8),
            build_residual_block(8),
            build_multi_conv_block(8, 8),
            nn.UpsamplingNearest2d(scale_factor=2.0),
        )

        self.f3_block1 = nn.Sequential(
            build_multi_conv_block(5, 8),
            build_multi_conv_block(8, 8),
            build_residual_block(8),
            build_residual_block(8),
            build_residual_block(8),
            build_multi_conv_block(8, 8),
            nn.UpsamplingNearest2d(scale_factor=4.0),
        )

        self.f3_block2 = nn.Sequential(
            nn.Conv2d(16, 8, 1, 1),
            build_multi_conv_block(8, 8),
        )

        self.output_block = nn.Sequential(
            build_residual_block(8),
            build_multi_conv_block(8, 8),
        )

    def forward(self, x, dcp, smoke):
        # [B, 3, H, W]
        # [B, 1, H, W]
        # [B, 1, H, W]

        _, _, h, w = x.shape

        x = torch.cat([x, dcp, smoke], dim=1)

        f1b1 = self.f1_block1(x)
        f2b = self.f2_block(F.interpolate(x, scale_factor=0.5, mode="bilinear"))
        f3b1 = self.f3_block1(F.interpolate(x, scale_factor=0.25, mode="bilinear"))

        if f2b.size()[2:] != f1b1.size()[2:]:
            f2b = F.interpolate(f2b, f1b1.size()[2:], mode="bilinear")

        f1b2 = self.f1_block2(f1b1 + f2b)

        if f1b2.size()[2:] != f3b1.size()[2:]:
            f1b2 = F.interpolate(f1b2, f3b1.size()[2:], mode="bilinear")

        f3b2 = self.f3_block2(torch.cat([f1b2, f3b1], dim=1))

        if f3b2.size()[2:] != f1b2.size()[2:]:
            f3b2 = F.interpolate(f3b2, f1b2.size()[2:], mode="bilinear")

        output = self.output_block(f1b2 + f3b2)

        if output.size()[2:] != x.size()[2:]:
            output = F.interpolate(output, x.size()[2:], mode="bilinear")

        return output


def build_gd_encoder_block(in_dim, out_dim):
    block = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, 3, 2, padding=1),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(inplace=True),
    )
    return block


def build_gd_decoder_block(in_dim, out_dim, act_f=True, bn=True):
    block = nn.Sequential(
        nn.UpsamplingNearest2d(scale_factor=2.0),
        nn.Conv2d(in_dim, out_dim, 3, 1, padding=1),
        nn.BatchNorm2d(out_dim) if bn else nn.Identity(),
        nn.ReLU(inplace=True) if act_f else nn.Identity(),
    )
    return block


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.skip_con = True

        self.en_block1 = build_gd_encoder_block(11, 64)
        self.en_block2 = build_gd_encoder_block(64, 128)
        self.en_block3 = build_gd_encoder_block(128, 256)
        self.en_block4 = build_gd_encoder_block(256, 512)
        self.en_block5 = build_gd_encoder_block(512, 512)

        self.bottle_neck = nn.Conv2d(512, 512, 1, 1)

        if self.skip_con:
            self.de_block5 = build_gd_decoder_block(512 * 2, 512)
            self.de_block4 = build_gd_decoder_block(512 * 2, 256)
            self.de_block3 = build_gd_decoder_block(256 * 2, 128)
            self.de_block2 = build_gd_decoder_block(128 * 2, 64)
            self.de_block1 = build_gd_decoder_block(64 * 2, 3, act_f=False, bn=False)
        else:
            self.de_block5 = build_gd_decoder_block(512, 512)
            self.de_block4 = build_gd_decoder_block(512, 256)
            self.de_block3 = build_gd_decoder_block(256, 128)
            self.de_block2 = build_gd_decoder_block(128, 64)
            self.de_block1 = build_gd_decoder_block(64, 3, act_f=False, bn=False)

        self.output_layer = nn.Sequential(
            nn.Sigmoid(),
        )

    def forward(self, x):
        # [B, 11, H, W]

        en1 = self.en_block1(x)
        en2 = self.en_block2(en1)
        en3 = self.en_block3(en2)
        en4 = self.en_block4(en3)
        en5 = self.en_block5(en4)

        neck = self.bottle_neck(en5)

        if self.skip_con:
            if neck.size()[2:] != en5.size()[2:]:
                neck = F.interpolate(neck, en5.size()[2:], mode="bilinear")
            de4 = self.de_block5(torch.cat([neck, en5], dim=1))
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
        else:
            de4 = self.de_block5(en5)
            de3 = self.de_block4(de4)
            de2 = self.de_block3(de3)
            de1 = self.de_block2(de2)
            de0 = self.de_block1(de1)

        output = self.output_layer(de0)

        if output.size()[2:] != x.size()[2:]:
            output = F.interpolate(output, x.size()[2:], mode="bilinear")

        return output


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.D = nn.Sequential(
            build_gd_encoder_block(3, 64),
            build_gd_encoder_block(64, 128),
            build_gd_encoder_block(128, 256),
            build_gd_encoder_block(256, 512),
            build_gd_encoder_block(512, 512),
        )

        self.output_layer = nn.Sequential(
            nn.Sigmoid(),
        )

    def forward(self, x):
        # [B, 3, H, W]

        x1 = self.D(x)

        x1 = x1.mean(dim=[-1, -2], keepdim=True)

        output = self.output_layer(x1)

        return output


class MARSGAN(BaseModel):
    def __init__(self):
        super().__init__()

        self.SSN = SSN()
        self.MSF = MSF()
        self.Gs = Generator()
        self.Gsl = Generator()
        self.Ds = Discriminator()
        self.Dsl = Discriminator()

        self.frozen_module_list = nn.ModuleList([])
        self.trained_module_list = nn.ModuleList([self.SSN, self.MSF, self.Gsl, self.Dsl, self.Gs, self.Ds])
        self.param_module_list = nn.ModuleList([self.Gsl])

        self.opt1_module = nn.ModuleList([self.SSN, self.MSF, self.Gsl, self.Gs])
        self.opt2_module = nn.ModuleList([self.Dsl, self.Ds])

        self.device = "cpu"

    def config_opt(self, opt_config):
        opt_list = [
            torch.optim.Adam(self.opt1_module.parameters(), lr=opt_config["lr"], betas=(opt_config["beta1"], opt_config["beta2"])),
            torch.optim.Adam(self.opt2_module.parameters(), lr=opt_config["lr"], betas=(opt_config["beta1"], opt_config["beta2"])),
        ]
        return opt_list

    def forward(self, x, sl2s=False):  # default is smoke to smoke_less, s2sl
        # [B, 3, H, W]

        smoke_pre = self.SSN(x)

        smoke_dcp = get_dcp_tensor(x)

        smoke_inter = torch.minimum(smoke_pre.detach(), smoke_dcp)

        msf_output = self.MSF(x, smoke_dcp, smoke_inter)

        x_msf = torch.cat([x, msf_output], dim=1)

        if not sl2s:
            de_smoke_x = self.Gsl(x_msf)
        else:
            de_smoke_x = self.Gs(x_msf)

        return de_smoke_x, smoke_pre

    def get_loss_d(self, d_x_gt, d_x, d_de_smoke_x, d_en_smoke_x):
        ones_tensor = torch.ones_like(d_x_gt, device=self.device)
        zeros_tensor = torch.zeros_like(d_x_gt, device=self.device)

        l_d_sl = F.binary_cross_entropy(d_x_gt, ones_tensor) + F.binary_cross_entropy(d_de_smoke_x, zeros_tensor)
        l_d_s = F.binary_cross_entropy(d_x, ones_tensor) + F.binary_cross_entropy(d_en_smoke_x, zeros_tensor)

        l_d = l_d_sl + l_d_s
        return l_d

    def get_loss_g(self, d_de_smoke_x, d_en_smoke_x):
        ones_tensor = torch.ones_like(d_de_smoke_x, device=self.device)
        l_g = F.binary_cross_entropy(d_de_smoke_x, ones_tensor) + F.binary_cross_entropy(d_en_smoke_x, ones_tensor)
        return l_g

    def get_loss_lambda(self, x, x_gt, smoke_pre, smoke_gt, de_smoke_x, en_smoke_x, gt_gs_gsl, x_gsl_gs):
        # [B, 3, H, W]
        # [B, 3, H, W]
        # [B, 1, H, W]

        dcp = get_dcp_tensor(x)

        l1_d = x_gt.sub(x).abs()
        dcp_smoke_inter = torch.minimum(dcp, smoke_pre).detach()
        l_sp_mask = torch.where(smoke_gt > 0, dcp_smoke_inter, 0.2)
        l_sp = l1_d.mul(l_sp_mask).mean()
        # l_sp = 0.0

        dcp_x_gt = get_dcp_tensor(x_gt)
        dcp_de_smoke_x = get_dcp_tensor(de_smoke_x)
        l_dcp = F.l1_loss(dcp_de_smoke_x, dcp_x_gt)

        x_gt_u = x_gt.mean(dim=[-1, -2])
        x_gt_u += 1e-6
        x_gt_v = x_gt.var(dim=[-1, -2])
        x_gt_v += 1e-6

        q_x_gt = x_gt_v / x_gt_u

        de_smoke_x_u = de_smoke_x.mean(dim=[-1, -2])
        de_smoke_x_u += 1e-6
        de_smoke_x_v = de_smoke_x.var(dim=[-1, -2])
        de_smoke_x_v += 1e-6

        q_de_smoke_x = de_smoke_x_v / de_smoke_x_u

        cef = q_de_smoke_x.add(1e-6) / q_x_gt.add(1e-6)
        l_ce = cef.add(1e-6).log().add(1e-6).reciprocal().abs().mean()

        l_cyc = F.l1_loss(gt_gs_gsl, x_gt) + F.l1_loss(x_gsl_gs, x)

        return l_sp, l_dcp, l_cyc, l_ce

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
            "Grad": 0,
            "Loss": 0,
            "SSN": 0,
            "D": 0,
            "G": 0,
            "Lambda": 0,
        }

        if only_dict:
            return loss_dict

        x = batch_dict["x"].to(self.device)
        gt = batch_dict["gt"].to(self.device)
        smoke_gt = batch_dict["smoke"].to(self.device)

        de_smoke_x, smoke_pre_1 = self.forward_dict(batch_dict)
        loss_ssn_1 = F.l1_loss(smoke_pre_1, smoke_gt)

        en_smoke_x, smoke_pre_0 = self.forward(gt, sl2s=True)
        loss_ssn_0 = smoke_pre_0.abs().mean()
        loss_ssn = loss_ssn_1 + loss_ssn_0

        smoke_pre = smoke_pre_1.detach()

        d_x_gt = self.Dsl(gt)
        d_x = self.Ds(x)
        d_de_smoke_x = self.Dsl(de_smoke_x)
        d_en_smoke_x = self.Ds(en_smoke_x)

        loss_d = self.get_loss_d(d_x_gt, d_x, d_de_smoke_x, d_en_smoke_x)

        loss_g = self.get_loss_g(d_de_smoke_x, d_en_smoke_x)

        x_gsl_gs, _ = self.forward(de_smoke_x.detach(), sl2s=True)
        gt_gs_gsl, _ = self.forward(en_smoke_x.detach())

        l_sp, l_dcp, l_cyc, l_ce, l_1 = self.get_loss_lambda(x, gt, smoke_pre, smoke_gt, de_smoke_x, en_smoke_x, gt_gs_gsl, x_gsl_gs)

        loss_sum = loss_ssn + loss_d + loss_g + l_cyc + l_sp + l_dcp + l_ce
        loss_grad = loss_ssn + loss_d + loss_g + l_cyc * 0.5 + l_sp * 0.8 + l_dcp * 0.4 + l_ce * 0.2

        loss_dict["Grad"] = loss_grad
        loss_dict["Loss"] = loss_sum
        loss_dict["SSN"] = loss_ssn
        loss_dict["D"] = loss_d
        loss_dict["G"] = loss_g
        loss_dict["Lambda"] = l_cyc + l_sp + l_dcp + l_ce

        return loss_dict


class MARSGAN_SurgiATM(MARSGAN):
    def __init__(self, dc_window_size=15, dc_bias=0):
        super().__init__()

        self.surgi_atm = SurgiATM(dc_window_size=dc_window_size, dc_bias=dc_bias)

    def forward(self, x, sl2s=False):  # default is smoke to smoke_less, s2sl
        # [B, 3, H, W]

        pre_smoke = self.SSN(x)

        smoke_dcp = get_dcp_tensor(x)

        smoke_inter = torch.minimum(pre_smoke.detach(), smoke_dcp)

        msf_output = self.MSF(x, smoke_dcp, smoke_inter)

        x_msf = torch.cat([x, msf_output], dim=1)

        if not sl2s:
            output = self.Gsl(x_msf)
            pre_gt, dcs, sdc = self.surgi_atm(x, output)
        else:
            pre_gt = self.Gs(x_msf)
            dcs, sdc = None, None

        return pre_gt, dcs, sdc, pre_smoke

    def get_loss_dict(self, batch_dict, only_dict=False):
        loss_dict = {
            "Grad": 0,
            "Loss": 0,
            "SSN": 0,
            "D": 0,
            "G": 0,
            "Lambda": 0,
        }

        if only_dict:
            return loss_dict

        x = batch_dict["x"].to(self.device)
        gt = batch_dict["gt"].to(self.device)
        smoke_gt = batch_dict["smoke"].to(self.device)

        de_smoke_x, smoke_pre_1 = self.forward_dict(batch_dict)
        loss_ssn_1 = F.l1_loss(smoke_pre_1, smoke_gt)

        en_smoke_x, _, _, smoke_pre_0 = self.forward(gt, sl2s=True)
        loss_ssn_0 = smoke_pre_0.abs().mean()
        loss_ssn = loss_ssn_1 + loss_ssn_0

        smoke_pre = smoke_pre_1.detach()

        d_x_gt = self.Dsl(gt)
        d_x = self.Ds(x)
        d_de_smoke_x = self.Dsl(de_smoke_x)
        d_en_smoke_x = self.Ds(en_smoke_x)

        loss_d = self.get_loss_d(d_x_gt, d_x, d_de_smoke_x, d_en_smoke_x)

        loss_g = self.get_loss_g(d_de_smoke_x, d_en_smoke_x)

        x_gsl_gs, _, _, _ = self.forward(de_smoke_x, sl2s=True)
        gt_gs_gsl, _, _, _ = self.forward(en_smoke_x)

        l_sp, l_dcp, l_cyc, l_ce = self.get_loss_lambda(x, gt, smoke_pre, smoke_gt, de_smoke_x, en_smoke_x, gt_gs_gsl, x_gsl_gs)

        loss_sum = loss_ssn + loss_d + loss_g + l_cyc + l_sp + l_dcp + l_ce
        loss_grad = loss_ssn + loss_d + loss_g + l_cyc * 0.5 + l_sp * 0.8 + l_dcp * 0.4 + l_ce + 0.2

        loss_dict["Grad"] = loss_grad
        loss_dict["Loss"] = loss_sum
        loss_dict["SSN"] = loss_ssn
        loss_dict["D"] = loss_d
        loss_dict["G"] = loss_g
        loss_dict["Lambda"] = l_cyc + l_sp + l_dcp + l_ce

        return loss_dict

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
