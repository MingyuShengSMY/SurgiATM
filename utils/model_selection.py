import torch

from model import BasedModel
from model.AODNet import AODNet, AODNet_SurgiATM
from model.CGANDC import CGANDC, CGANDC_SurgiATM
from model.GCANet import GCANet, GCANet_SurgiATM
from model.LGUTransformer import LGUTransformer, LGUTransformer_SurgiATM
from model.DCP import DCP
from model.DesmokeGCN import DesmokeGCN, DesmokeGCN_SurgiATM
from model.MARSGAN import MARSGAN, MARSGAN_SurgiATM
from model.MSBDN import MSBDN, MSBDN_SurgiATM
from model.RSTN import RSTN, RSTN_SurgiATM
from model.SSIM_PAN import SSIM_PAN, SSIM_PAN_SurgiATM
from model.SVPNet import SVPNet, SVPNet_SurgiATM


def select_model(method_name: str, config_method=None) -> BasedModel:

    if "_SurgiATM" in method_name:
        dc_window_size = config_method.args.dc_window_size
        dc_bias = config_method.args.dc_bias
    else:
        dc_window_size = 15
        dc_bias = 0.1

    if method_name == "DesmokeGCN":
        model = DesmokeGCN()
    elif method_name == "DesmokeGCN_SurgiATM":
        model = DesmokeGCN_SurgiATM(dc_window_size, dc_bias)
    elif method_name == "GCANet":
        model = GCANet()
    elif method_name == "GCANet_SurgiATM":
        model = GCANet_SurgiATM(dc_window_size, dc_bias)
    elif method_name == "MARSGAN":
        model = MARSGAN()
    elif method_name == "MARSGAN_SurgiATM":
        model = MARSGAN_SurgiATM(dc_window_size, dc_bias)
    elif method_name == "CGANDC":
        model = CGANDC()
    elif method_name == "CGANDC_SurgiATM":
        model = CGANDC_SurgiATM(dc_window_size, dc_bias)
    elif method_name == "MSBDN":
        model = MSBDN()
    elif method_name == "MSBDN_SurgiATM":
        model = MSBDN_SurgiATM(dc_window_size, dc_bias)
    elif method_name == "AODNet":
        model = AODNet()
    elif method_name == "AODNet_SurgiATM":
        model = AODNet_SurgiATM(dc_window_size, dc_bias)
    elif method_name == "LGUTransformer":
        model = LGUTransformer()
    elif method_name == "LGUTransformer_SurgiATM":
        model = LGUTransformer_SurgiATM(dc_window_size, dc_bias)
    elif method_name == "SSIM_PAN":
        model = SSIM_PAN()
    elif method_name == "SSIM_PAN_SurgiATM":
        model = SSIM_PAN_SurgiATM(dc_window_size, dc_bias)
    elif method_name == "RSTN":
        model = RSTN()
    elif method_name == "RSTN_SurgiATM":
        model = RSTN_SurgiATM(dc_window_size, dc_bias)
    elif method_name == "SVPNet":
        model = SVPNet()
    elif method_name == "SVPNet_SurgiATM":
        model = SVPNet_SurgiATM(dc_window_size, dc_bias)
    elif method_name == "DCP":
        model = DCP()
    else:
        raise ValueError(f"Unknown method name: '{method_name}'")

    return model


def select_opt(opt_config, parameters):
    opts = []
    for opt_c_i in opt_config:
        opt_name = opt_c_i["name"]
        if opt_name == "Adam":
            opt = torch.optim.Adam(parameters, lr=opt_c_i["lr"], betas=(opt_c_i["beta1"], opt_c_i["beta2"]))
        else:
            raise ValueError(f"Unknown optimizer name: '{opt_name}'")
        opts.append(opt)

    return opts
