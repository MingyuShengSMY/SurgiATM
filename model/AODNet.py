"""
Following the work:

Li, B., Peng, X., Wang, Z., Xu, J. and Feng, D., 2017. Aod-net: All-in-one dehazing network.
In Proceedings of the IEEE international conference on computer vision (pp. 4770-4778).

From:
https://github.com/Boyiliee/AOD-Net/tree/master
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from model.BasedModel import BaseModel
import timm
import torchvision.transforms.functional as TF

from model.SurgiATM import SurgiATM
from utils.tool_functions import get_dcp_tensor


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=7, stride=1, padding=3)
        self.conv5 = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        cat1 = torch.cat((x1, x2), 1)
        x3 = F.relu(self.conv3(cat1))
        cat2 = torch.cat((x2, x3), 1)
        x4 = F.relu(self.conv4(cat2))
        cat3 = torch.cat((x1, x2, x3, x4), 1)
        k = self.conv5(cat3)

        return k


class AODNet(BaseModel):
    def __init__(self):
        super().__init__()

        self.net = Net()

        self.frozen_module_list = nn.ModuleList([])
        self.trained_module_list = nn.ModuleList([self.net])
        self.param_module_list = nn.ModuleList([self.net])

        self.device = "cpu"

    def forward(self, x):
        # [B, 3, H, W]

        k = self.net(x)

        k = F.relu(k)

        output = F.relu(k * x - k + 1.0)

        return output, k

    def forward_dict(self, batch_dict):
        x = batch_dict["x"].to(self.device)
        # [B, 3, H, W]

        de_smoke_x, k = self.forward(x)

        k = k.div(k.amax(dim=[-3, -2, -1], keepdim=True).maximum(torch.tensor([1.0], device=k.device)))

        batch_dict["pre_gt"] = de_smoke_x.detach()

        batch_dict["vis"] = {}

        batch_dict["vis"]["k"] = k.detach()

        return de_smoke_x

    def get_loss_dict(self, batch_dict, only_dict=False):
        loss_dict = {
            "Grad": 0,
            "Loss": 0,
        }

        if only_dict:
            return loss_dict

        x = batch_dict["x"].to(self.device)
        gt = batch_dict["gt"].to(self.device)

        de_smoke_x = self.forward_dict(batch_dict)

        loss_l2 = F.mse_loss(gt, de_smoke_x)

        loss_grad = loss_l2
        loss_sum = loss_l2

        loss_dict["Grad"] = loss_grad
        loss_dict["Loss"] = loss_sum

        return loss_dict


class AODNet_SurgiATM(AODNet):
    def __init__(self, dc_window_size=15, dc_bias=0):
        super().__init__()

        self.surgi_atm = SurgiATM(dc_window_size=dc_window_size, dc_bias=dc_bias)

    def forward(self, x):
        # [B, 3, H, W]

        output = self.net(x)

        output = F.sigmoid(output)

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

