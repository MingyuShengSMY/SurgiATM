"""
Referred from https://github.com/guptapraful/niqe
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms.functional as tv_tf
import numpy as np
import scipy.misc
import scipy.io
from os.path import dirname
from os.path import join
import scipy
import scipy.ndimage
import scipy.special


def gen_gauss_window(lw, sigma):
    sd = np.float32(sigma)
    lw = int(lw)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd *= sd
    for ii in range(1, lw + 1):
        tmp = np.exp(-0.5 * np.float32(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
    return weights


class NIQE:
    def __init__(self):
        self.patch_size = 96

        gamma_range = np.arange(0.2, 10, 0.001)
        a = scipy.special.gamma(2.0 / gamma_range)
        a *= a
        b = scipy.special.gamma(1.0 / gamma_range)
        c = scipy.special.gamma(3.0 / gamma_range)
        prec_gammas = a / (b * c)
        self.gamma_range = torch.from_numpy(gamma_range)  # [9800]
        self.prec_gammas = torch.from_numpy(prec_gammas).unsqueeze(0).unsqueeze(0)  # [1, 1, 9800]

        module_path = dirname(__file__)
        self.param = scipy.io.loadmat(join(module_path, 'niqe_image_params.mat'))

        self.pop_mu = torch.from_numpy(np.ravel(self.param["pop_mu"])).unsqueeze(0)  # [1, 36]
        self.pop_cov = torch.from_numpy(self.param["pop_cov"]).unsqueeze(0)  # [1, 36, 36]

    def get_anchor_mv(self):
        m = self.pop_mu[0]
        v = self.pop_cov[0]
        v_idx = torch.triu_indices(row=v.shape[0], col=v.shape[1])
        v = v[v_idx[0], v_idx[1]]
        mv = torch.cat([m, v], dim=-1)
        return mv

    def score(self, img: torch.Tensor, return_feat=False):
        # [N, H, W]  0~255

        n, h, w = img.shape
        device = img.device
        self.pop_mu = self.pop_mu.to(device)
        self.pop_cov = self.pop_cov.to(device)
        self.gamma_range = self.gamma_range.to(device)
        self.prec_gammas = self.prec_gammas.to(device)

        feats = self.get_patches_test_features(img)  # [N, PN, 36]

        _, pn, _ = feats.shape

        sample_mu = feats.mean(dim=1)  # [N, 36]

        feats_transpose = feats.transpose(-1, -2)  # [N, 36, PN]
        ft_mean = feats_transpose.mean(dim=-1, keepdim=True)  # [N, 36, 1]
        ft_c = feats_transpose - ft_mean  # [N, 36, PN]
        sample_cov = torch.bmm(ft_c, ft_c.transpose(-1, -2)) / (pn - 1)  # [N, 36, 36]

        X = sample_mu - self.pop_mu
        X = X.unsqueeze(-1)  # [N, 36, 1]
        X_T = X.transpose(-1, -2)  # [N, 1, 36]
        covmat = ((self.pop_cov + sample_cov) / 2.0)  # [N, 36, 36]
        pinvmat = torch.linalg.pinv(covmat)  # [N, 36, 36]
        niqe_score = torch.sqrt(X_T @ pinvmat @ X).flatten(-2, -1)  # [N]

        if not return_feat:
            return niqe_score
        else:
            return niqe_score, (feats, ft_mean, sample_cov)

    def compute_image_mscn_transform(self, image: torch.Tensor):

        mu_image = tv_tf.gaussian_blur(image, kernel_size=[7, 7], sigma=[7.0/6.0, 7.0/6.0])
        var_image = tv_tf.gaussian_blur(image.square(), kernel_size=[7, 7], sigma=[7.0/6.0, 7.0/6.0])

        var_image = var_image.sub(mu_image.square()).abs().sqrt()

        mscn = (image - mu_image)/(var_image + 1e-8)

        return mscn, var_image, mu_image

    def aggd_features(self, imdata: torch.Tensor):
        # [N, PN, P, P]
        # flatten imdata
        imdata = imdata.flatten(-2, -1)  # [N, PN, P*P]
        imdata2 = imdata.square()
        left_index = torch.argwhere(imdata < 0)[:, -1]
        left_data = imdata2[:, :, left_index]  # [N, PN, ~]
        right_index = torch.argwhere(imdata >= 0)[:, -1]
        right_data = imdata2[:, :, right_index]  # [N, PN, ~]
        left_mean_sqrt = 0
        right_mean_sqrt = 0
        if left_data.shape[-1] > 0:
            left_mean_sqrt = left_data.mean(dim=-1).sqrt()
        if right_data.shape[-1] > 0:
            right_mean_sqrt = right_data.mean(dim=-1).sqrt()  # [N, PN]

        gamma_hat = torch.where(right_mean_sqrt==0, torch.inf, left_mean_sqrt / right_mean_sqrt)  # [N, NP]
        # solve r-hat norm

        imdata2_mean = imdata2.mean(dim=-1)  # [N, PN]
        r_hat = torch.where(imdata2_mean==0, torch.inf, (imdata.abs().mean(dim=-1).square()) / imdata2_mean)  # [N, NP]

        # rhat_norm = r_hat * (((math.pow(gamma_hat, 3) + 1) * (gamma_hat + 1)) / math.pow(math.pow(gamma_hat, 2) + 1, 2))
        rhat_norm = r_hat * (((gamma_hat.pow(3) + 1) * (gamma_hat + 1)) / gamma_hat.pow(2).add(1).pow(2))  # [N, PN]

        # solve alpha by guessing values that minimize ro
        pos = torch.argmin((self.prec_gammas.sub(rhat_norm.unsqueeze(-1)).square()), dim=-1)  # [N, PN]
        alpha = self.gamma_range[pos]  # [N, PN]

        gam1 = torch.exp(torch.lgamma(1.0 / alpha))  # [N, PN]
        gam2 = torch.exp(torch.lgamma(2.0 / alpha))
        gam3 = torch.exp(torch.lgamma(3.0 / alpha))

        aggdratio = gam1.sqrt().div(gam3.sqrt())
        bl = aggdratio * left_mean_sqrt
        br = aggdratio * right_mean_sqrt

        # mean parameter
        N = (br - bl) * (gam2 / gam1)  # *aggdratio
        return (alpha, N, bl, br, left_mean_sqrt, right_mean_sqrt)

    def paired_product(self, new_im):
        # [N, PN, P, P]
        shift1 = torch.roll(new_im, 1, dims=-1)
        shift2 = torch.roll(new_im, 1, dims=-2)
        shift3 = torch.roll(shift2, 1, dims=-1)
        shift4 = torch.roll(shift2, -1, dims=-1)

        H_img = shift1 * new_im
        V_img = shift2 * new_im
        D1_img = shift3 * new_im
        D2_img = shift4 * new_im

        return (H_img, V_img, D1_img, D2_img)

    def _niqe_extract_subband_feats(self, mscncoefs):
        # alpha_m,  = extract_ggd_features(mscncoefs)
        alpha_m, N, bl, br, lsq, rsq = self.aggd_features(mscncoefs)  # [N, PN] ...
        pps1, pps2, pps3, pps4 = self.paired_product(mscncoefs)
        alpha1, N1, bl1, br1, lsq1, rsq1 = self.aggd_features(pps1)
        alpha2, N2, bl2, br2, lsq2, rsq2 = self.aggd_features(pps2)
        alpha3, N3, bl3, br3, lsq3, rsq3 = self.aggd_features(pps3)
        alpha4, N4, bl4, br4, lsq4, rsq4 = self.aggd_features(pps4)

        blr = (bl + br) / 2.0

        feats = torch.stack((alpha_m, blr,
                             alpha1, N1, bl1, br1,   # (V)
                             alpha2, N2, bl2, br2,   # (H)
                             alpha3, N3, bl3, bl3,   # (D1)
                             alpha4, N4, bl4, bl4),  # (D2)
                            dim=-1
                            )  # [N, NP, 18]
        return feats

    def extract_on_patches(self, img: torch.Tensor, patch_size) -> torch.Tensor:
        img = img.unsqueeze(1)
        n, _, h, w = img.shape

        patches = F.unfold(img, kernel_size=patch_size, stride=patch_size)  # [N, 1*P*P, P_N]
        patches = patches.permute(0, 2, 1).reshape(n, -1, patch_size, patch_size)  # [N, P_N, P, P]
        patch_features = self._niqe_extract_subband_feats(patches)  # [N, PN, 18]

        return patch_features

    def get_patches_test_features(self, img: torch.Tensor):
        n, h, w = img.shape
        if h < self.patch_size or w < self.patch_size:
            print("Input image is too small")
            exit(0)

        # ensure that the patch divides evenly into img
        # hoffset = (h % self.patch_size)
        # woffset = (w % self.patch_size)
        #
        # if hoffset > 0:
        #     img = img[:-hoffset, :]
        # if woffset > 0:
        #     img = img[:, :-woffset]

        img = img.to(torch.float32)

        h2, w2 = h // 2, w // 2
        img2 = tv_tf.resize(img, size=[h2, w2])

        mscn1, var, mu = self.compute_image_mscn_transform(img)
        mscn2, _, _ = self.compute_image_mscn_transform(img2)

        feats_lvl1 = self.extract_on_patches(mscn1, self.patch_size)
        feats_lvl2 = self.extract_on_patches(mscn2, self.patch_size // 2)

        feats = torch.cat([feats_lvl1, feats_lvl2], dim=-1)  # [N, PN, 36]

        return feats




