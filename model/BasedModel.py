import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fvcore.nn import FlopCountAnalysis


class BaseModel(nn.Module):

    trained_module_list: nn.ModuleList
    frozen_module_list: nn.ModuleList
    param_module_list: nn.ModuleList
    device: str

    def forward(self, *args):
        pass

    def forward_dict(self, *arg):
        pass

    def get_loss(self, *arg):
        pass

    def get_loss_dict(self, *arg):
        pass

    def config_opt(self, opt_config):
        if len(list(self.trained_module_list.parameters())) == 0:
            return []

        parameters = self.trained_module_list.parameters()

        opt_list = [torch.optim.Adam(parameters, lr=opt_config["lr"], betas=(opt_config["beta1"], opt_config["beta2"]))]
        return opt_list

    def opt_update(self, loss_dict, opts: list[torch.optim.Optimizer], *args):
        loss_grad = loss_dict.get("Grad") if loss_dict.get("Grad") is not None else loss_dict.get("Loss")
        for opt in opts:
            opt.zero_grad()
        loss_grad.backward()
        for opt in opts:
            opt.step()
            opt.zero_grad()

    def before_epoch(self, epoch, max_epoch):
        pass

    def __get_flops__(self, input_tensor):
        flops_counter = FlopCountAnalysis(self, input_tensor)
        flops_counter.unsupported_ops_warnings(False)
        flops_counter.uncalled_modules_warnings(False)
        flops = flops_counter.total()
        return flops

    def get_flops(self):
        input_tensor = torch.randn(2, 3, 256, 256, device=self.device)
        # some methods need at least 2 samples because of BN layer at neck place.
        flops = self.__get_flops__(input_tensor) / 2
        return flops

    def get_flops_video(self, length=1):
        # True the flops mark, because in the real practical using, some input information can be previously obtained
        input_tensor = (torch.randn(2, length, 3, 256, 256, device=self.device), True)
        flops = self.__get_flops__(input_tensor) / 2
        return flops


class ChannelLayerNorm(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.layer_norm = nn.LayerNorm(num_channels)

    def forward(self, x):
        # [B, C, H, W]

        x_permuted = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x_normalized = self.layer_norm(x_permuted)

        return x_normalized.permute(0, 3, 1, 2)  # [B, C, H, W]

