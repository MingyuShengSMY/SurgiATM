"""
Following the work:

Wang, F., Sun, X. and Li, J., 2023. Surgical smoke removal via residual Swin transformer network. International
Journal of Computer Assisted Radiology and Surgery, 18(8), pp.1417-1427.
"""
from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from torchvision.models.swin_transformer import SwinTransformerBlock, SwinTransformer
import torchvision.models as TVM

from model.BasedModel import BaseModel
import timm
import torchvision.transforms.functional as TF

from model.SurgiATM import SurgiATM


class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg16 = TVM.vgg16(weights=TVM.VGG16_Weights.IMAGENET1K_V1).features

        # Default layers: relu1_2, relu2_2, relu3_4, relu4_4
        self.feature_layers = [3, 8, 15]

        self.feature_extractor = nn.Sequential(
            *[self.vgg16[i] for i in range(max(self.feature_layers) + 1)]
        )

    def forward(self, x):
        # [B, 3, H, W]
        feature_maps = []
        feat = x
        for i, layer in enumerate(self.feature_extractor):
            feat = layer(feat)
            if i in self.feature_layers:
                feature_maps.append(feat)

        return feature_maps


class RSTN(BaseModel):
    def __init__(self):
        super().__init__()

        self.FE = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.Conv2d(128, 180, 1, 1, 0),
        )

        self.FFE = nn.Sequential(
            SwinTransformerBlock(dim=180, num_heads=6, window_size=[8, 8], shift_size=[0, 0]),
            SwinTransformerBlock(dim=180, num_heads=6, window_size=[8, 8], shift_size=[4, 4]),
            SwinTransformerBlock(dim=180, num_heads=6, window_size=[8, 8], shift_size=[0, 0]),
            SwinTransformerBlock(dim=180, num_heads=6, window_size=[8, 8], shift_size=[4, 4]),
            SwinTransformerBlock(dim=180, num_heads=6, window_size=[8, 8], shift_size=[0, 0]),
            SwinTransformerBlock(dim=180, num_heads=6, window_size=[8, 8], shift_size=[4, 4]),
        )

        self.FR = nn.Sequential(
            nn.Conv2d(180, 128, 1, 1, 0),
            nn.UpsamplingNearest2d(scale_factor=2.0),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.UpsamplingNearest2d(scale_factor=2.0),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Sigmoid(),
        )

        self.VGG16 = VGG16()

        self.frozen_module_list = nn.ModuleList([self.VGG16])
        self.trained_module_list = nn.ModuleList([self.FE, self.FFE, self.FR])
        self.param_module_list = nn.ModuleList([self.FE, self.FFE, self.FR])

        self.device = "cpu"

    def forward(self, x):
        # [B, 3, H, W]

        feat1 = self.FE(x)

        feat1_t = feat1.permute(0, 2, 3, 1)  # [b, h, w, c]

        feat2_t = self.FFE(feat1_t)

        feat2 = feat2_t.permute(0, 3, 1, 2)

        feat1 = feat1 + feat2

        output = self.FR(feat1)

        if output.size()[2:] != x.size()[2:]:
            output = F.interpolate(output, x.size()[2:], mode="bilinear")

        return output

    def forward_dict(self, batch_dict):
        x = batch_dict["x"].to(self.device)
        # [B, 3, H, W]

        de_smoke_x = self.forward(x)
        batch_dict["pre_gt"] = de_smoke_x.detach()

        return de_smoke_x

    def get_char_loss(self, de_smoke_x, gt):
        loss = torch.sqrt(torch.square(de_smoke_x.sub(gt)).add(1e-6)).mean()
        return loss

    def get_per_loss(self, de_smoke_x, gt):
        feat_maps_de = self.VGG16(de_smoke_x)
        feat_maps_gt = self.VGG16(gt)

        loss = 0.0
        for feat_de, feat_gt in zip(feat_maps_de, feat_maps_gt):
            loss += F.mse_loss(feat_de, feat_gt)

        return loss

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
        # loss_ssim = ms_ssim_bchw_tensor_01(de_smoke_x, gt, norm=True)
        loss_ssim = 1 - self.__msssim__(de_smoke_x, gt, normalize=True)
        return loss_ssim

    def get_loss_dict(self, batch_dict, only_dict=False):
        loss_dict = {
            "Grad": 0,
            "Loss": 0,
            "CHA": 0,
            "SS": 0,
            "PER": 0,
        }

        if only_dict:
            return loss_dict

        x = batch_dict["x"].to(self.device)
        gt = batch_dict["gt"].to(self.device)

        de_smoke_x = self.forward_dict(batch_dict)

        loss_char = self.get_char_loss(de_smoke_x, gt)

        loss_per = self.get_per_loss(de_smoke_x, gt)
        # loss_per = 0.0

        loss_ssim = self.get_loss_ssim(de_smoke_x, gt)
        # loss_ssim = 0.0

        loss_grad = loss_char + loss_per * 0.001 + loss_ssim * 0.5
        loss_sum = loss_char + loss_per + loss_ssim

        loss_dict["Grad"] = loss_grad
        loss_dict["Loss"] = loss_sum
        loss_dict["CHA"] = loss_char
        loss_dict["PER"] = loss_per
        loss_dict["SS"] = loss_ssim

        return loss_dict


class RSTN_SurgiATM(RSTN):
    def __init__(self, dc_window_size=15, dc_bias=0):
        super().__init__()

        self.surgi_atm = SurgiATM(dc_window_size=dc_window_size, dc_bias=dc_bias)

    def forward(self, x):
        # [B, 3, H, W]

        feat1 = self.FE(x)

        feat1_t = feat1.permute(0, 2, 3, 1)  # [b, h, w, c]

        feat2_t = self.FFE(feat1_t)

        feat2 = feat2_t.permute(0, 3, 1, 2)

        feat1 = feat1 + feat2

        output = self.FR(feat1)

        if output.size()[2:] != x.size()[2:]:
            output = F.interpolate(output, x.size()[2:], mode="bilinear")

        # output = (output + 1.0) / 2.0

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

