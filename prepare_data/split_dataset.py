import os
import glob
import shutil

import cv2
import numpy as np
import sys
import argparse
from sklearn.model_selection import KFold
import re

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset_name')

args = parser.parse_args()

RANDOM_SEED = 2025

TARGET_DATASET = args.dataset_name

root_dir = ".."

dataset_dir = f"{root_dir}/dataset/{TARGET_DATASET}"

assert TARGET_DATASET is not None, f"Use '--dataset_name [DATASET_NAME]'"


def split_image_data():
    x_path_list = []
    gt_path_list = []
    x_path_list += sorted(glob.glob(f"*/*.png", root_dir=f"{dataset_dir}/X"))
    gt_path_list += sorted(glob.glob(f"*/*.png", root_dir=f"{dataset_dir}/GT"))
    x_path_list += sorted(glob.glob(f"*/*.jpg", root_dir=f"{dataset_dir}/X"))
    gt_path_list += sorted(glob.glob(f"*/*.jpg", root_dir=f"{dataset_dir}/GT"))
    x_path_list += sorted(glob.glob(f"*/*.PNG", root_dir=f"{dataset_dir}/X"))
    gt_path_list += sorted(glob.glob(f"*/*.PNG", root_dir=f"{dataset_dir}/GT"))
    x_path_list += sorted(glob.glob(f"*/*.JPG", root_dir=f"{dataset_dir}/X"))
    gt_path_list += sorted(glob.glob(f"*/*.JPG", root_dir=f"{dataset_dir}/GT"))
    x_path_list = set(x_path_list)
    gt_path_list = set(gt_path_list)

    if len(x_path_list) * len(gt_path_list) > 0:
        path_list = list(x_path_list.intersection(gt_path_list))
    else:
        path_list = list(x_path_list) + list(gt_path_list)

    os.makedirs(f"{dataset_dir}/data_split", exist_ok=True)

    image_list = path_list

    image_list.sort()

    image_n = len(image_list)

    te_n = max(int(image_n * 0.2), 1)
    tr_n = image_n - te_n

    assert tr_n > 0

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    for fold_id, (tr_idx, te_idx) in enumerate(kf.split(image_list), 1):
        fold_path_i = f"{dataset_dir}/data_split/{fold_id}"
        os.makedirs(fold_path_i, exist_ok=True)

        te_sample_list = np.array(image_list)[te_idx].tolist()
        va_sample_list = np.array(image_list)[te_idx].tolist()
        tr_sample_list = np.array(image_list)[tr_idx].tolist()

        with open(f"{fold_path_i}/train.txt", "w") as f:
            f.write("\n".join(tr_sample_list))
        with open(f"{fold_path_i}/val.txt", "w") as f:
            f.write("\n".join(va_sample_list))
        with open(f"{fold_path_i}/test.txt", "w") as f:
            f.write("\n".join(te_sample_list))

        if fold_id == 1:
            with open(f"{dataset_dir}/data_split/1.txt", "w") as f:
                f.write("\n".join(tr_sample_list[1:2]))

            all_sample_list = tr_sample_list + va_sample_list + te_sample_list

            with open(f"{dataset_dir}/data_split/all.txt", "w") as f:
                f.write("\n".join(all_sample_list))


def split_video_data():
    x_path_list = []
    gt_path_list = []
    x_path_list += sorted(glob.glob(f"*/*.png", root_dir=f"{dataset_dir}/X"))
    gt_path_list += sorted(glob.glob(f"*/*.png", root_dir=f"{dataset_dir}/GT"))
    x_path_list += sorted(glob.glob(f"*/*.jpg", root_dir=f"{dataset_dir}/X"))
    gt_path_list += sorted(glob.glob(f"*/*.jpg", root_dir=f"{dataset_dir}/GT"))
    x_path_list += sorted(glob.glob(f"*/*.PNG", root_dir=f"{dataset_dir}/X"))
    gt_path_list += sorted(glob.glob(f"*/*.PNG", root_dir=f"{dataset_dir}/GT"))
    x_path_list += sorted(glob.glob(f"*/*.JPG", root_dir=f"{dataset_dir}/X"))
    gt_path_list += sorted(glob.glob(f"*/*.JPG", root_dir=f"{dataset_dir}/GT"))
    x_path_list = set(x_path_list)
    gt_path_list = set(gt_path_list)

    if len(x_path_list) * len(gt_path_list) > 0:
        path_list = list(x_path_list.intersection(gt_path_list))
    else:
        path_list = list(x_path_list) + list(gt_path_list)

    os.makedirs(f"{dataset_dir}/data_split", exist_ok=True)

    video_list = list(set([p.split("/")[0] for p in path_list]))

    video_list.sort()

    video_n = len(video_list)

    te_n = max(int(video_n * 0.2), 1)
    tr_n = video_n - te_n

    assert tr_n > 0

    k_fold_n = min(video_n, 5)

    kf = KFold(n_splits=k_fold_n, shuffle=True, random_state=RANDOM_SEED)

    for fold_id, (tr_idx, te_idx) in enumerate(kf.split(video_list), 1):
        fold_path_i = f"{dataset_dir}/data_split/{fold_id}"
        os.makedirs(fold_path_i, exist_ok=True)

        te_video = np.array(video_list)[te_idx].tolist()
        va_video = np.array(video_list)[te_idx].tolist()
        tr_video = np.array(video_list)[tr_idx].tolist()

        tr_sample_list = []
        va_sample_list = []
        te_sample_list = []

        video_dict = {k: [] for k in video_list}

        for sample_name in path_list:
            video_name = sample_name.split("/")[0]

            if video_dict.get(video_name) is not None:
                video_dict[video_name].append(sample_name)

        for video_name in video_list:
            sample_list_i = video_dict[video_name]
            sample_list_i = sorted(sample_list_i, key=lambda x: int(re.findall(r"\d+", x)[-1]))

            if video_name in tr_video:
                tr_sample_list += sample_list_i
            if video_name in va_video:
                va_sample_list += sample_list_i
            if video_name in te_video:
                te_sample_list += sample_list_i

        with open(f"{fold_path_i}/train.txt", "w") as f:
            f.write("\n".join(tr_sample_list))
        with open(f"{fold_path_i}/val.txt", "w") as f:
            f.write("\n".join(va_sample_list))
        with open(f"{fold_path_i}/test.txt", "w") as f:
            f.write("\n".join(te_sample_list))

        if fold_id == 1:
            with open(f"{dataset_dir}/data_split/1.txt", "w") as f:
                f.write("\n".join(tr_sample_list[1:2]))

            # all_sample_list = tr_sample_list + va_sample_list + te_sample_list
            all_sample_list = tr_sample_list + va_sample_list

            with open(f"{dataset_dir}/data_split/all.txt", "w") as f:
                f.write("\n".join(all_sample_list))


def copy_cholec80_data_split():
    smoky_dataset_dir = f"{root_dir}/dataset/Cholec80_smoky"
    clean_dataset_dir = f"{root_dir}/dataset/Cholec80_clean"

    smoky_data_split_folder = f"{smoky_dataset_dir}/data_split"
    clean_data_split_folder = f"{clean_dataset_dir}/data_split"

    shutil.copytree(clean_data_split_folder, smoky_data_split_folder, dirs_exist_ok=True)


def main():
    split_video_data()


if __name__ == '__main__':
    main()

