import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms.functional as tv_tf
from torchvision.transforms._functional_tensor import _rgb2hsv
import numpy as np
import scipy.misc
import scipy.io
from os.path import dirname
from os.path import join
import scipy
import scipy.ndimage
import scipy.special
import tqdm
import cv2
from scipy.io import loadmat
from utils.tool_functions import debug_cv_show
import kornia


class CIEDE2000:
    def __init__(self, patch_size=8):
        self.e1 = 25 ** 7

        self.K_L = 1
        self.K_C = 1
        self.K_H = 1

    def score(self, pre: torch.Tensor, gt: torch.Tensor):
        # [B, 3, H, W]   0 ~ 255

        pre = pre / 255.0
        gt = gt / 255.0

        Lab_pre = kornia.color.rgb_to_lab(pre)
        Lab_gt = kornia.color.rgb_to_lab(gt)

        L_pre, a_pre, b_pre = Lab_pre.unbind(dim=1)  # [B, H, W]
        L_gt, a_gt, b_gt = Lab_gt.unbind(dim=1)  # [B, H, W]

        C_pre_ab = (a_pre.square() + b_pre.square()).sqrt()
        C_gt_ab = (a_gt.square() + b_gt.square()).sqrt()

        C_ab = (C_pre_ab + C_gt_ab) * 0.5

        G = 0.5 * (1 - torch.sqrt(C_ab.pow(7) / (C_ab.pow(7) + self.e1)))

        a_pre_ = (1 + G) * a_pre
        a_gt_ = (1 + G) * a_gt

        C_pre = (a_pre_.square() + b_pre.square()).sqrt()
        C_gt = (a_gt_.square() + b_gt.square()).sqrt()

        h_pre = torch.where((a_pre_ == b_pre) * (a_pre_ == 0), 0, torch.atan2(b_pre, a_pre_))
        h_gt = torch.where((a_gt_ == b_gt) * (a_gt_ == 0), 0, torch.atan2(b_gt, a_gt_))

        delta_L = L_gt - L_pre
        delta_C = C_gt - C_pre
        CC = C_pre * C_gt
        hh = h_gt - h_pre
        hh_abs = hh.abs()
        delta_h = hh
        CC0_mask = CC == 0
        CC1_mask = ~CC0_mask

        delta_h[CC0_mask] = 0

        delta_h = torch.where(CC1_mask * (hh_abs <= torch.pi), hh, delta_h)
        delta_h = torch.where(CC1_mask * (hh > torch.pi), hh - 2 * torch.pi, delta_h)
        delta_h = torch.where(CC1_mask * (hh < -torch.pi), hh + 2 * torch.pi, delta_h)

        delta_H = 2 * CC.sqrt() * torch.sin(delta_h / 2)

        L = (L_pre + L_gt) / 2
        C = (C_pre + C_gt) / 2

        hh_add = h_pre + h_gt

        h = hh_add
        h = torch.where(CC1_mask * (hh_abs <= torch.pi), hh_add / 2, h)
        h = torch.where(CC1_mask * (hh_abs > torch.pi) * hh_add < 2 * torch.pi, (hh_add + 2 * torch.pi) / 2, h)
        h = torch.where(CC1_mask * (hh_abs > torch.pi) * hh_add >= 2 * torch.pi, (hh_add - 2 * torch.pi) / 2, h)

        T = 1 - 0.17 * torch.cos(h - torch.pi / 6) + 0.24 * torch.cos(2 * h) + 0.32 * torch.cos(3 * h + torch.pi / 30) - 0.2 * torch.cos(4 * h - torch.pi * 63 / 180)

        delta_theta = 30 * torch.exp(-((h - torch.pi * 275 / 180).square() / 625))

        R_C = 2 * torch.sqrt(C.pow(7) / (C.pow(7) + self.e1))

        S_L = 1 + (0.015 * (L - 50).square()) / (20 + (L - 50).square()).sqrt()

        S_C = 1 + 0.045 * C

        S_H = 1 + 0.015 * C * T

        R_T = - torch.sin(2 * delta_theta) * R_C

        score = (
                (delta_L / S_L).square() +
                (delta_C / S_C).square() +
                (delta_H / S_H).square() +
                (delta_C / S_C) * (delta_H / S_H) * R_T
                 ).sqrt()

        score = score.mean(dim=[-1, -2]).flatten()

        return score


def test():
    obj = CIEDE2000()

    img_smoky = tv.io.read_image("test_image2.JPG").unsqueeze(0)
    img_clean = tv.io.read_image("test_image1.png").unsqueeze(0)

    img_smoky = tv_tf.resize(img_smoky, size=[256, 256])
    img_clean = tv_tf.resize(img_clean, size=[256, 256])

    fade_score_smoky1 = obj.score(img_smoky, img_clean)[0].item()
    fade_score_smoky2 = obj.score(img_smoky, (img_smoky + img_clean)/2)[0].item()
    fade_score_smoky3 = obj.score(img_smoky, img_smoky)[0].item()

    print("Score:", fade_score_smoky1)
    print("Score:", fade_score_smoky2)
    print("Score:", fade_score_smoky3)


if __name__ == '__main__':
    test()
