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


def torch_nan_var(input, dim=None, keepdim=False, unbiased=True):
    mask = ~torch.isnan(input)
    count = mask.sum(dim=dim, keepdim=True)
    count = count.clamp(min=1)

    input_zeroed = torch.where(mask, input, 0)
    mean = input_zeroed.sum(dim=dim, keepdim=True) / count

    sq_diff = torch.where(mask, (input - mean) ** 2, 0)
    if unbiased:
        denom = count - 1
        denom = denom.clamp(min=1)
    else:
        denom = count
    var = sq_diff.sum(dim=dim, keepdim=True) / denom

    if not keepdim:
        var = var.squeeze(dim)

    return var


def get_ce_gaussian_kernel():
    sigma = 3.25
    bos = 3

    filter_size = bos * sigma

    x = torch.arange(-filter_size, filter_size + 1e-9, 1)

    gau = 1/((2 * torch.pi) ** 0.5 * sigma) * torch.exp((x ** 2)/(-2 * sigma ** 2))
    gau = gau / gau.sum()
    gx = (x ** 2 / sigma ** 4 - 1 / sigma ** 2) * gau
    gx = gx - gx.sum() / x.nelement()
    gx = gx / gx.mul(x).mul(x).mul(0.5).sum()

    return gx


class FADE:
    def __init__(self, patch_size=8):
        self.patch_size = patch_size

        self.smoky_mu = np.zeros(shape=[1, 12])
        self.smoky_cov = np.eye(12)
        self.clean_mu = np.zeros(shape=[1, 12])
        self.clean_cov = np.eye(12)

        file_cwd = "./" + os.path.split(os.path.relpath(__file__, start=os.getcwd()))[0]

        smoky_feature_mat_path = "natural_foggy_image_features_ps8.mat"
        clean_feature_mat_path = "natural_fogfree_image_features_ps8.mat"

        smoke_feat_mat = loadmat(f"{file_cwd}/{smoky_feature_mat_path}")
        clean_feat_mat = loadmat(f"{file_cwd}/{clean_feature_mat_path}")

        self.smoky_mu = smoke_feat_mat["mu_foggyparam"]
        self.smoky_cov = smoke_feat_mat["cov_foggyparam"]
        self.clean_mu = clean_feat_mat["mu_fogfreeparam"]
        self.clean_cov = clean_feat_mat["cov_fogfreeparam"]

        self.smoky_mu = torch.from_numpy(self.smoky_mu).unsqueeze(0).unsqueeze(0) # [1, 1, 12]
        self.smoky_cov = torch.from_numpy(self.smoky_cov).unsqueeze(0).unsqueeze(0)  # [1, 1, 12, 12]

        self.clean_mu = torch.from_numpy(self.clean_mu).unsqueeze(0).unsqueeze(0)  # [1, 1, 12]
        self.clean_cov = torch.from_numpy(self.clean_cov).unsqueeze(0).unsqueeze(0)  # [1, 1, 12, 12]

        self.h_w_cache = (0, 0)

        self.ce_gau_kernel = get_ce_gaussian_kernel().reshape(1, 1, -1, 1)  # [1, 1, k, 1]

    def score(self, img: torch.Tensor, return_feat=False):
        # [N, 3, H, W]  0~255

        # img = img / 255.0

        img = img.to(torch.float32)

        n, _, h, w = img.shape
        device = img.device

        # return torch.zeros([n], device=device)

        h = int(np.round(h / self.patch_size) * self.patch_size)
        w = int(np.round(w / self.patch_size) * self.patch_size)

        img = tv_tf.resize(img, size=[h, w], interpolation=tv_tf.InterpolationMode.BILINEAR)

        self.ce_gau_kernel = self.ce_gau_kernel.to(device)
        self.smoky_mu = self.smoky_mu.to(device)
        self.smoky_cov = self.smoky_cov.to(device)
        self.clean_mu = self.clean_mu.to(device)
        self.clean_cov = self.clean_cov.to(device)

        img_patch_feature = self.get_patches_feature(img)  # [N, pn, 12]
        img_patch_feature = torch.log(img_patch_feature + 1)

        img_mu, img_cov = self.get_mvg(img_patch_feature)  # [N, pn, 1, 12], [N, pn, 1, 1]

        d_f = torch.sqrt(
            (img_mu - self.clean_mu) @
            torch.linalg.pinv((img_cov + self.clean_cov) / 2) @
            (img_mu - self.clean_mu).transpose(-1, -2)
        ).mean(dim=1)

        d_ff = torch.sqrt(
            (img_mu - self.smoky_mu) @
            torch.linalg.pinv((img_cov + self.smoky_cov) / 2) @
            (img_mu - self.smoky_mu).transpose(-1, -2)
        ).mean(dim=1)

        fade_score = d_f.div(d_ff + 1).flatten()

        if not return_feat:
            return fade_score
        else:
            return fade_score, (img_mu, img_cov)

    def patch_img(self, img: torch.Tensor):
        n, d, h, w = img.shape
        img_patched = F.unfold(img, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2).unflatten(-1, (d, self.patch_size, self.patch_size))  # [N, pn, 3, h, w]
        return img_patched

    def de_patch_img(self, img_patched: torch.Tensor):
        n, pn, d, h, w = img_patched.shape
        img_patched = img_patched.flatten(-3, -1).transpose(-1, -2)  # [n, d*h*w, pn]
        img = F.fold(img_patched, kernel_size=self.patch_size, stride=self.patch_size, output_size=self.h_w_cache).transpose(1, 2).unflatten(-1, (d, self.patch_size, self.patch_size))  # [N, pn, 3, h, w]
        return img

    def get_patches_feature(self, img: torch.Tensor):

        n, d, h, w = img.shape

        self.h_w_cache = (h, w)

        img_patched = self.patch_img(img)  # [n, pn, d, h, w]

        pn = img_patched.shape[1]

        img_gray = tv_tf.rgb_to_grayscale(img)
        img_gray_patched = tv_tf.rgb_to_grayscale(img_patched).squeeze(2)
        img_r_patched = img_patched[:, :, 0, ...]
        img_g_patched = img_patched[:, :, 1, ...]
        img_b_patched = img_patched[:, :, 2, ...]
        img_rg_patched = img_r_patched - img_g_patched
        img_by_patched = 0.5 * (img_r_patched + img_g_patched) - img_b_patched

        img_dc_patched = img_patched.min(dim=-3, keepdim=True)[0] / 255.0

        img_hsv_s = _rgb2hsv(img)[:, 1:2, :, :]

        img_mscn, img_mscn_mu, img_mscn_sigma, img_mscn_cv = self.cal_img_mscn(img_gray)  # [n, 1, h, w]
        img_mscn_patched = self.patch_img(img_mscn)
        img_f1 = torch_nan_var(img_mscn_patched, dim=[-1, -2])  # [n, pn, 1]

        img_mscn_roll = torch.roll(img_mscn, dims=-1, shifts=1)
        img_mscn_v_pair = img_mscn * img_mscn_roll
        img_mscn_v_l, img_mscn_v_r = self.cal_vp_mscn(img_mscn_v_pair)
        img_mscn_v_l_patched, img_mscn_v_r_patched = self.patch_img(img_mscn_v_l), self.patch_img(img_mscn_v_r)
        img_f2 = torch_nan_var(img_mscn_v_l_patched, dim=[-1, -2])
        img_f3 = torch_nan_var(img_mscn_v_r_patched, dim=[-1, -2])

        img_mscn_mu_patched = self.patch_img(img_mscn_mu)
        img_f4 = img_mscn_mu_patched.mean(dim=[-1, -2])

        img_mscn_cv_patched = self.patch_img(img_mscn_cv)
        img_f5 = img_mscn_cv_patched.mean(dim=[-1, -2])

        ce_gray, ce_by, ce_rg = self.get_ce(img)
        ce_gray = self.patch_img(ce_gray)
        ce_by = self.patch_img(ce_by)
        ce_rg = self.patch_img(ce_rg)
        img_f6 = ce_gray.mean(dim=[-1, -2])
        img_f7 = ce_by.mean(dim=[-1, -2])
        img_f8 = ce_rg.mean(dim=[-1, -2])

        img_f9 = self.cal_ie(img_gray_patched)  # [n, pn, 1]
        img_f10 = img_dc_patched.mean(dim=[-1, -2])

        img_hsv_s_patched = self.patch_img(img_hsv_s)
        img_f11 = img_hsv_s_patched.mean(dim=[-1, -2])

        img_f12 = self.cal_cf(img_by_patched, img_rg_patched)  # [n, pn, 1]

        img_feat = torch.cat([
            img_f1, img_f2, img_f3, img_f4, img_f5, img_f6, img_f7, img_f8, img_f9, img_f10, img_f11, img_f12,
        ], dim=-1)  # [n, pn, 12]

        return img_feat

    def cal_img_mscn(self, img: torch.Tensor):
        # [n, d, h, w]

        mu_img = tv_tf.gaussian_blur(img, kernel_size=[7, 7], sigma=[7/6, 7/6])

        sigma_img = tv_tf.gaussian_blur(img.sub(mu_img).square(), kernel_size=[7, 7], sigma=[7/6, 7/6]).sqrt()

        mscn = (img - mu_img)/(sigma_img + 1)

        img_cv = sigma_img.div(mu_img + 1e-6)

        return mscn, mu_img, sigma_img, img_cv

    def cal_vp_mscn(self, img: torch.Tensor):

        img_l = img.clone()
        img_r = img.clone()

        img_l[img_l > 0] = torch.nan
        img_r[img_r < 0] = torch.nan

        return img_l, img_r

    def cal_ce(self, img: torch.Tensor, t):
        ss = 0.1
        pad = self.ce_gau_kernel.nelement() // 2
        pad_list_x = (0, 0, pad, pad - (1 if pad % 2 == 0 else 0))
        pad_list_y = (pad, pad - (1 if pad % 2 == 0 else 0), 0, 0)

        img_gray_pad_x = F.pad(img, pad_list_x, mode="replicate")
        img_gray_pad_y = F.pad(img, pad_list_y, mode="replicate")

        cx_gray = F.conv2d(img_gray_pad_x, self.ce_gau_kernel)
        cy_gray = F.conv2d(img_gray_pad_y, self.ce_gau_kernel.transpose(-1, -2))
        c_gray = cx_gray.square().add(cy_gray.square()).sqrt()
        r_gray = (c_gray * c_gray.max()) / (c_gray + (c_gray.max() * ss)) - t
        r_gray = r_gray.clamp(1e-7)

        return r_gray

    def cal_ie(self, img: torch.Tensor):

        n, pn, h, w = img.shape

        hw = h * w

        img = img.clamp(0, 255).to(torch.long).flatten(-2, -1)  # [N, pn, hw]

        img_oh = F.one_hot(img, 256)  # [N, pn, hw, 256]

        hist_count = img_oh.sum(dim=-2)  # [N, pn, 256]

        hist = hist_count / hw

        ie = - torch.sum(hist * hist.add(1e-6).log2(), dim=-1, keepdim=True)  # [N, pn, 1]

        return ie

    def cal_cf(self, img_yb: torch.Tensor, img_rg: torch.Tensor):

        u_yb = img_yb.mean(dim=[-1, -2])
        s_yb = img_yb.std(dim=[-1, -2])

        u_rg = img_rg.mean(dim=[-1, -2])
        s_rg = img_rg.std(dim=[-1, -2])

        cf = s_yb.square().add(s_rg.square()).sqrt() + 0.3 * u_yb.square().add(u_rg.square()).sqrt()

        return cf.unsqueeze(-1)

    def get_mvg(self, img_feat: torch.Tensor):
        # [n, pn, 12]

        img_mu = img_feat.unsqueeze(-2)
        img_cov = torch_nan_var(img_feat, dim=-1, keepdim=True).unsqueeze(-1)  # [n, pn, 1, 1]

        return img_mu, img_cov

    def get_ce(self, img: torch.Tensor):
        ss = 0.1
        t1 = 9.225496406318721e-004 * 255
        t2 = 8.969246659629488e-004 * 255
        t3 = 2.069284034165411e-004 * 255

        img_r = img[:, 0:1, ...]
        img_g = img[:, 1:2, ...]
        img_b = img[:, 2:3, ...]

        img_gray = img_r * 0.299 + img_g * 0.587 + img_b * 0.114
        img_yb = 0.5 * (img_r + img_g) - img_b
        img_rg = img_r - img_g

        r_gray = self.cal_ce(img_gray, t1)
        r_yb = self.cal_ce(img_yb, t2)
        r_rg = self.cal_ce(img_rg, t3)

        return r_gray, r_yb, r_rg

    def get_mvg_distance(self, img_patch_features):
        # [n, pn, 12]

        n, pn, fd = img_patch_features.shape

        img_mu = img_patch_features  # [n, pn, 12]
        img_cov = torch_nan_var(img_patch_features, dim=-1, keepdim=True).unsqueeze(-1)  # [n, pn, 1, 1]

        mu_matrix = self.clean_mu - img_mu  # [n, pn, 12]

        cov_matrix = (img_cov + self.clean_cov) / 2




def get_images(data_dir):

    video_list = os.listdir(data_dir)

    video_range = tqdm.tqdm(
        range(len(video_list)),
        position=0,
        leave=False,
        desc="Video",
    )

    img_list = []

    for video_idx in video_range:
        video_name = video_list[video_idx]

        video_dir = f"{data_dir}/{video_name}"

        img_name_list = os.listdir(video_dir)

        # np.random.shuffle(img_name_list)

        random_img_name_list = img_name_list[:10]
        # random_img_name_list = img_name_list

        for img_name in random_img_name_list:

            img_path = f"{video_dir}/{img_name}"

            img = tv.io.read_image(img_path, mode=tv.io.ImageReadMode.RGB)
            # img = tv.io.read_image(img_path, mode=tv.io.ImageReadMode.RGB).to("cuda")

            img_list.append(img.unsqueeze(0))

            # break

    return img_list


def cal_all_features(img_list):
    all_feature = []

    fade_obj = FADE(patch_size=64)

    img_range = tqdm.tqdm(
        range(len(img_list)),
        position=0,
        leave=False,
        desc="Img",
    )

    for img_idx in img_range:

        img = img_list[img_idx]

        img = img / 255.0

        n, _, h, w = img.shape

        h = int(np.round(h / fade_obj.patch_size) * fade_obj.patch_size)
        w = int(np.round(w / fade_obj.patch_size) * fade_obj.patch_size)

        img = tv_tf.resize(img, size=[h, w], interpolation=tv_tf.InterpolationMode.BILINEAR)

        img_patch_feature = fade_obj.get_patches_feature(img)  # [N, pn, 12]

        all_feature.append(img_patch_feature.flatten(0, 1))

    all_feature = torch.cat(all_feature, dim=0)

    img_mu, img_cov = fade_obj.get_mvg(all_feature)  # [1, 12], [12, 12]

    return img_mu, img_cov


def test():
    fade_obj = FADE()

    img_smoky = tv.io.read_image("test_image2.JPG").unsqueeze(0)
    img_clean = tv.io.read_image("test_image1.png").unsqueeze(0)

    fade_score_smoky = fade_obj.score(img_smoky)[0].item()
    fade_score_clean = fade_obj.score(img_clean)[0].item()

    print("Smoky:", fade_score_smoky)
    print("Clean:", fade_score_clean)


if __name__ == '__main__':
    test()
