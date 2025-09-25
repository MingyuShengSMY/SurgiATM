import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as tv_tf


class SurgiATM(nn.Module):
    def __init__(self, dc_window_size=15, dc_bias=0.1, w=0.95):
        super().__init__()
        self.dc_wz = dc_window_size
        self.w = w
        self.dc_bias = dc_bias

    def __get_dc__(self, x: torch.Tensor):
        # [..., C, H, W]

        pd = self.dc_wz // 2

        dc_pixel_wise, _ = x.min(dim=-3, keepdim=True)  # [..., 1, H, W]

        if self.dc_wz > 1:
            dc_pixel_wise_pad = F.pad(dc_pixel_wise, (pd, pd, pd, pd), mode="replicate")

            dc = - F.max_pool2d(-dc_pixel_wise_pad, kernel_size=self.dc_wz, stride=1, padding=0)
        else:
            dc = dc_pixel_wise

        return dc  # [..., 1, H, W]

    def __get_a__(self, x: torch.Tensor, dc: torch.Tensor):
        b, c, h, w = x.shape
        hw = h * w

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

        return atm_light

    def __get_pw_a__(self, x: torch.Tensor, dc: torch.Tensor):
        b, c, h, w = x.shape
        hw = h * w

        pd = self.dc_wz // 2

        x_pad = F.pad(x, (pd, pd, pd, pd), mode="replicate")

        atm_light = F.max_pool2d(x_pad, kernel_size=self.dc_wz, stride=1)

        return atm_light

    def avg_pool2d(self, x: torch.Tensor):
        return F.avg_pool2d(x, 31, 1, 31 // 2, count_include_pad=False)

    def guided_filter_refine(self, img, t):

        self.refine_avg_pool = self.avg_pool2d

        img = tv_tf.rgb_to_grayscale(img)

        mean_img = self.refine_avg_pool(img)
        mean_t = self.refine_avg_pool(t)
        mean_img_t = self.refine_avg_pool(t * img)
        cov_img_t = mean_img_t - mean_img * mean_t

        mean_img_img = self.refine_avg_pool(img * img)
        var_img = mean_img_img - mean_img * mean_img

        a = cov_img_t / (var_img + 1e-3)
        b = mean_t - a * mean_img

        mean_a = self.refine_avg_pool(a)
        mean_b = self.refine_avg_pool(b)

        q = mean_a * img + mean_b

        return q

    def forward(self, smoky_image: torch.Tensor, model_output: torch.Tensor):
        """
        reconstruct smokeless images
        :param smoky_image:  shape is [..., C, H, W], C is the color channel, usually C=1 or C=3. The value range
        should be [0, 1].
        :param model_output:  The same shape as the smoky_image. The value range should be [0, 1].
        :return: pre_clean_image, scaled_dark_channel, dark_channel_scalar
        """

        dc = self.__get_dc__(smoky_image)  # [..., 1, H, W]

        # dc = 1 - self.guided_filter_refine(smoky_image, 1 - dc)   # This visualization is better ?

        # dc = dc.clamp(min=0, max=1)

        # atm_light = self.__get_a__(smoky_image, dc)
        # atm_light = self.__get_pw_a__(smoky_image, dc)

        dc_atm = dc
        dc_atm_tuned = (self.dc_bias + dc_atm) / (self.dc_bias + 1)

        dc_scalar = model_output

        scaled_dc = dc_atm_tuned * dc_scalar

        pre_clean_image = smoky_image - scaled_dc

        return pre_clean_image, dc_scalar, scaled_dc
