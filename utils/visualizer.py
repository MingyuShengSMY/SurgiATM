import torchvision.transforms.functional as TF
import torchvision.io as TVIO
import torch
import os
import sys

sys.path.append("..")


def vis_for_loop(pre_gt, pre_gt_dir, origin_shape, sample_name, fixed_sample_name):
    pre_gt = torch.clamp(pre_gt, 0, 1)

    n = len(pre_gt)
    if fixed_sample_name is not None:
        n = 1

    for i in range(n):
        pre_gt_i = pre_gt[i]

        origin_shape_i = eval(origin_shape[i])
        sample_name_i = sample_name[i]
        if fixed_sample_name is not None:
            sample_name_i = fixed_sample_name

        pre_gt_i = TF.resize(pre_gt_i, size=origin_shape_i, interpolation=TF.InterpolationMode.BILINEAR).mul(255).to(
            torch.uint8)
        pre_gt_save_path_i = f"{pre_gt_dir}/{sample_name_i}"
        pre_gt_save_dir_i = os.path.split(pre_gt_save_path_i)[0]
        os.makedirs(pre_gt_save_dir_i, exist_ok=True)
        TVIO.write_png(pre_gt_i, filename=pre_gt_save_path_i)


def visualize_dict(batch_dict: dict, output_dir: str, fixed_sample_name=None):
    if batch_dict.get("gt") is not None:
        gt = batch_dict.get("gt")
        gt_dir = f"{output_dir}/gt"
        gt_mark = True
    else:
        gt = None
        gt_mark = False

    pre_gt: torch.Tensor = batch_dict.get("pre_gt").cpu()
    pre_gt_dir = f"{output_dir}/pre_gt"

    origin_shape = batch_dict.get("origin_shape")
    sample_name = batch_dict.get("sample_name")

    if len(pre_gt.shape) >= 5:
        video_mark = True
        if gt_mark:
            gt = gt[:, -1, ...]
        pre_gt = pre_gt[:, -1, ...]
        sample_name = sample_name[-1]
        origin_shape = origin_shape[-1]
    else:
        video_mark = False
    if gt_mark:
        gt = gt.clamp(0, 1)
    pre_gt = pre_gt.clamp(0, 1)

    vis_for_loop(pre_gt, pre_gt_dir, origin_shape, sample_name, fixed_sample_name)

    batch_vis = batch_dict.get("vis")

    if batch_vis is not None:
        for k in batch_vis:
            k_dir = f"{output_dir}/{k}"
            v = batch_vis.get(k).cpu()

            if video_mark:
                v = v[:, -1, ...]

            vis_for_loop(v, k_dir, origin_shape, sample_name, fixed_sample_name)








