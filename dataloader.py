import os.path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import re
import cv2
import torchvision
import torchvision.io as tv_io


class MyDataset(Dataset):
    def __init__(self, dataset_name: str, dataset_dir: str, sample_txt_path, data_aug_mark=False,
                 train_video_length=5, train_mode=False, gt_mark=True, video_based=True,
                 input_size=None, method_name=None):
        if input_size is None:
            input_size = [256, 256]

        if isinstance(train_video_length, int):
            self.train_video_length = train_video_length
        elif isinstance(train_video_length, str):
            self.train_video_length = 1
        else:
            raise ValueError(f"Wrong 'train_video_length' in config file")

        self.train_mode = train_mode
        self.video_based = video_based
        self.input_size = input_size
        self.gt_mark = gt_mark

        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.data_aug_mark = data_aug_mark and train_mode

        with open(sample_txt_path, "r") as f:
            self.sample_name_list = f.read().split("\n")

        self.x_dir = f"{self.dataset_dir}/{self.dataset_name}/X"
        if not self.gt_mark:
            self.gt_dir = None
        else:
            self.gt_dir = f"{self.dataset_dir}/{self.dataset_name}/GT"

        self.smoke_dir = f"{self.dataset_dir}/{self.dataset_name}/smoke"

        if not os.path.exists(self.smoke_dir):
            self.smoke_dir = None

        self.video_frame_dict = {}

        for sample_name in self.sample_name_list:
            video_name = sample_name.split("/")[0]

            if self.video_frame_dict.get(video_name) is not None:
                self.video_frame_dict[video_name].append(sample_name)
            else:
                self.video_frame_dict[video_name] = [sample_name]

        self.video_name_set = list(self.video_frame_dict.keys())

        for v in self.video_frame_dict:
            self.video_frame_dict[v] = sorted(self.video_frame_dict[v], key=lambda x: int(re.findall(r"\d+", x)[-1]))

        pass

    def __len__(self):
        return len(self.sample_name_list)

    def read_a_frame(self, sample_name):
        if self.gt_dir is not None:
            gt = tv_io.read_image(f"{self.gt_dir}/{sample_name}", mode=tv_io.ImageReadMode.RGB) / 255.0

            if self.input_size != -1:
                gt = TF.resize(gt, size=self.input_size, interpolation=TF.InterpolationMode.BILINEAR)
            if min(np.array(gt.size())[1:]) < 256:
                scale_factor = 256 / min(gt.size()[1:])
                gt = TF.resize(gt, size=list((np.array(gt.size())[1:] * scale_factor).astype(np.int32)), interpolation=TF.InterpolationMode.BILINEAR)

        else:
            gt = None
        if self.smoke_dir is not None:
            smoke = tv_io.read_image(f"{self.smoke_dir}/{sample_name}", mode=tv_io.ImageReadMode.GRAY) / 255.0
            if self.input_size != -1:
                smoke = TF.resize(smoke, size=self.input_size, interpolation=TF.InterpolationMode.BILINEAR)
            if min(smoke.size()[1:]) < 256:
                scale_factor = 256 / min(smoke.size()[1:])
                smoke = TF.resize(smoke, size=list((np.array(smoke.size())[1:] * scale_factor).astype(np.int32)), interpolation=TF.InterpolationMode.BILINEAR)
        else:
            smoke = None

        x = tv_io.read_image(f"{self.x_dir}/{sample_name}", mode=tv_io.ImageReadMode.RGB) / 255.0

        origin_shape = x.shape[1:]

        if self.input_size != -1:
            x = TF.resize(x, size=self.input_size, interpolation=TF.InterpolationMode.BILINEAR)
        if min(x.size()[1:]) < 256:
            scale_factor = 256 / min(x.size()[1:])
            x = TF.resize(x, size=list((np.array(x.size())[1:] * scale_factor).astype(np.int32)),
                              interpolation=TF.InterpolationMode.BILINEAR)

        return_dict = {}
        if gt is not None:
            return_dict["gt"] = gt
        if smoke is not None:
            return_dict["smoke"] = smoke

        return_dict["x"] = x
        return_dict["sample_name"] = sample_name
        return_dict["origin_shape"] = str(tuple(origin_shape))

        return return_dict

    def __getitem__(self, item):
        if not self.video_based:
            sample_dict = self.read_a_frame(sample_name=self.sample_name_list[item])
            return sample_dict
        else:
            sample_name = self.sample_name_list[item]
            video_name = sample_name.split("/")[0]
            # video_name = self.video_name_set[item]
            frame_list = self.video_frame_dict[video_name]

            sample_idx = frame_list.index(sample_name)
            start_idx = max(sample_idx - (self.train_video_length - 1), 0)

            frame_list = frame_list[start_idx: sample_idx + 1]
            if len(frame_list) < self.train_video_length:
                frame_list = [frame_list[0]] * (self.train_video_length - len(frame_list)) + frame_list

            video_dict = {
                "gt": [],
                "x": [],
                # "x_syn": [],
                "smoke": [],
                # "smoke_syn": [],
                "sample_name": [],
                "origin_shape": [],
            }

            if not self.gt_mark:
                video_dict.pop("gt")
                video_dict.pop("smoke")

            for frame_name in frame_list:
                frame_dict = self.read_a_frame(sample_name=frame_name)

                for k in frame_dict:
                    v = frame_dict[k]
                    if torch.is_tensor(v):
                        v = v.unsqueeze(0)

                    video_dict[k].append(v)

            if self.gt_mark:
                video_dict["gt"] = torch.cat(video_dict["gt"], dim=0)
            video_dict["x"] = torch.cat(video_dict["x"], dim=0)
            if self.gt_mark:
                video_dict["smoke"] = torch.cat(video_dict["smoke"], dim=0)
            video_dict["sample_name"] = video_dict["sample_name"]
            video_dict["origin_shape"] = video_dict["origin_shape"]

            return video_dict


def get_dataset(config, one_sample_overfitting=False, vis=False):
    dataset_dir = config.dataset.dir
    dataset_name = config.dataset.name
    dataset_fold_id = config.dataset.fold_id
    dataset_test_on_all_fold = config.dataset.test_on_all_fold
    gt_mark = config.dataset.gt_mark
    video_input = config.method.video_input
    method_name = config.method.name
    input_size = config.method.input_size
    train_video_length = config.method.train_video_length
    test_video_length = config.method.test_video_length

    if not one_sample_overfitting:
        if dataset_test_on_all_fold:  # exp on all data, mainly for DCP
            tr_txt_path = va_txt_path = te_txt_path = f"{dataset_dir}/{dataset_name}/data_split/all.txt"
        else:
            tr_txt_path = f"{dataset_dir}/{dataset_name}/data_split/{dataset_fold_id}/train.txt"
            va_txt_path = f"{dataset_dir}/{dataset_name}/data_split/{dataset_fold_id}/val.txt"
            te_txt_path = f"{dataset_dir}/{dataset_name}/data_split/{dataset_fold_id}/test.txt"
    else:
        tr_txt_path = va_txt_path = te_txt_path = f"{dataset_dir}/{dataset_name}/data_split/1.txt"

    tr_dataset = MyDataset(dataset_name=dataset_name, dataset_dir=dataset_dir, sample_txt_path=tr_txt_path,
                           data_aug_mark=False, train_video_length=train_video_length,
                           train_mode=True, gt_mark=gt_mark,
                           video_based=video_input, input_size=input_size, method_name=method_name)

    va_dataset = MyDataset(dataset_name=dataset_name, dataset_dir=dataset_dir, sample_txt_path=va_txt_path,
                           data_aug_mark=False, train_video_length=train_video_length,
                           train_mode=False, gt_mark=gt_mark,
                           video_based=video_input, input_size=input_size, method_name=method_name)

    te_dataset = MyDataset(dataset_name=dataset_name, dataset_dir=dataset_dir, sample_txt_path=te_txt_path,
                           data_aug_mark=False, train_video_length=test_video_length,
                           train_mode=False, gt_mark=gt_mark,
                           video_based=video_input, input_size=input_size, method_name=method_name)

    # te_dataset = MyDataset(dataset_name=dataset_name, dataset_dir=dataset_dir, sample_txt_path=te_txt_path,
    #                        data_aug_mark=False, train_video_length=test_video_length,
    #                        train_mode=False, gt_mark=gt_mark,
    #                        video_based=video_input, input_size=-1 if vis else input_size, method_name=method_name)

    return tr_dataset, va_dataset, te_dataset


def worker_init_fn(worker_id):
    # Set the seed for each worker, ensuring they all use the same base seed but with worker-specific offset
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def get_dataloader(config, tr_dataset, va_dataset, te_dataset, vis=False):
    method_name = config.method.name

    tr_dataloader = DataLoader(
        tr_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        worker_init_fn=worker_init_fn,
        # drop_last=True,
        pin_memory=True,
    )
    va_dataloader = DataLoader(
        va_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
    )

    te_dataloader = DataLoader(
        te_dataset,
        # batch_size=1 if vis else config.training.batch_size,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
    )

    return tr_dataloader, va_dataloader, te_dataloader


if __name__ == '__main__':
    my_dataset = MyDataset("WenyaoDataset", dataset_dir="dataset",
                           sample_txt_path="dataset/WenyaoDataset/data_split/train.txt", train_mode=True)

    my_dataset.__getitem__(0)
