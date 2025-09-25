import time

import numpy as np
import cv2
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
import sys


def debug_cv_show(img):
    cv2.imshow("debug", img)
    cv2.waitKey()
    cv2.destroyWindow("debug")


def wait():
    wait_mark = True
    while wait_mark:
        with open("wait.txt", "r") as f:
            a = f.read()

        try:
            a = int(a)
            wait_mark = a != 0
        except:
            wait_mark = False

        if wait_mark:
            torch.cuda.empty_cache()
        if wait_mark:
            time.sleep(1)


def min_max_normalization(data, dim=None):
    data_min = data.min(dim)
    data_max = data.max(dim)

    max_min = data_max - data_min

    normalized_data = (data - data_min) / (data_max - data_min)

    return normalized_data


def get_dcp_np(x, s=15):
    # [h, w, 3]
    min_c = np.min(x, axis=-1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (s, s))
    dcp = cv2.erode(min_c, kernel)
    return dcp


def get_dcp_tensor(x: torch.Tensor, s=15):
    # [b, 3, h, w]
    min_c = x.min(dim=1, keepdim=True)[0]

    pad_s = (s - 1) // 2

    min_c = F.pad(min_c, (pad_s, pad_s, pad_s, pad_s), mode="replicate")

    dcp = -F.max_pool2d(-min_c, kernel_size=s, stride=1, padding=0)
    return dcp


def get_normed_dcp_tensor(x: torch.Tensor, s=15):
    # [b, 3, h, w]

    b, c, h, w = x.shape
    hw = h * w

    min_c = x.min(dim=1, keepdim=True)[0]

    pad_s = (s - 1) // 2

    min_c = F.pad(min_c, (pad_s, pad_s, pad_s, pad_s), mode="replicate")

    dc = -F.max_pool2d(-min_c, kernel_size=s, stride=1, padding=0)

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

    dcp = dc / (atm_light + 1e-6)

    return dcp


def gaussian_kernel(kernel_size=11, sigma=1.5, channels=1):
    """Creates a 2D Gaussian kernel for convolution."""
    x = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    kernel_1d = gauss / gauss.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d.expand(channels, 1, kernel_size, kernel_size)  # Expand for convolution
    return kernel_2d


def ms_ssim_bchw_tensor_01(img1, img2, kernel_size=11, sigma=1.5, L=1.0, weights=None, levels=5, norm=False):
    # Ensure the inputs are float tensors
    img1 = img1.float()
    img2 = img2.float()

    # Default weights if not provided
    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]  # Default MS-SSIM weights
    weights = torch.tensor(weights).to(img1.device)

    msssim = []
    for i in range(levels):
        ssim_val = ssim_bchw_tensor_01(img1, img2, window_size=kernel_size, L=L, norm=norm)
        msssim.append(ssim_val)

        # Downsample by a factor of 2 for the next scale
        if i < levels - 1:  # No need to downsample at the last level
            img1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
            img2 = F.avg_pool2d(img2, kernel_size=2, stride=2)

    # Combine SSIM scores using the weights
    msssim = torch.stack(msssim)
    return (msssim ** weights).prod()


def ssim_bchw_tensor_01(img1, img2, window_size=11, size_average=True, L=1.0, norm=False):
    b, c, h, w = img1.shape

    kernel = gaussian_kernel(window_size, 1.5, c).to(img1.device)

    mu1 = F.conv2d(img1, kernel, padding=window_size // 2, groups=c)
    mu2 = F.conv2d(img2, kernel, padding=window_size // 2, groups=c)

    mu1_pow_2 = mu1.pow(2)
    mu2_pow_2 = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=window_size // 2, groups=c) - mu1_pow_2
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=window_size // 2, groups=c) - mu2_pow_2
    sigma12 = F.conv2d(img1 * img2, kernel, padding=window_size // 2, groups=c) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_pow_2 + mu2_pow_2 + C1) * (sigma1_sq + sigma2_sq + C2)
    if norm:
        ssim_map = (numerator + 1) / (denominator + 1)  # to avoid nan
    else:
        ssim_map = numerator / denominator  # to avoid nan

    return ssim_map.mean()


def generate_pseudo_smoke_mask(x, gt, s=15):
    dcp_x = get_dcp_np(x, s=s)
    dcp_gt = get_dcp_np(gt, s=s)

    mask = dcp_x - dcp_gt
    mask = np.clip(mask, 0, 1)

    mask = (mask * 255).astype(np.uint8)

    return mask


def seed_everything(seed=2025):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_model_parameters(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params
