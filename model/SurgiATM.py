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
    
    def forward(self, smoky_image: torch.Tensor, model_output: torch.Tensor):
        """
        reconstruct smokeless images
        :param smoky_image:  shape is [..., C, H, W], C is the color channel, usually C=1 or C=3. The value range
        should be [0, 1].
        :param model_output:  The same shape as the smoky_image. The value range should be [0, 1].
        :return: pre_clean_image, normalized_radiance, dc_rho
        """

        dc = self.__get_dc__(smoky_image)  # [..., 1, H, W]

        normalized_radiance = model_output  # \rho

        dc_rho = (self.dc_bias + dc) / (self.dc_bias + 1) * (1 - normalized_radiance)

        pre_clean_image = smoky_image - dc_rho

        return pre_clean_image, normalized_radiance, dc_rho
