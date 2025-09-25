# Run this with: blender --python path/to/this/file.py
# The blender version is 2.70
# The code is based on https://github.com/melights/cnn_desmoke/blob/master/render_smoke.py
import argparse
import glob
import os
from noise import pnoise2, pnoise3
import numpy as np
import cv2
import tqdm


RANDOM_SEED = 2025
np.random.seed(RANDOM_SEED)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset_name')

args = parser.parse_args()

# Change this to your project absolute path
# ROOT_PATH = "/data/Mingyu/SurgeryDeSmoking/MyProject/"

# TARGET_DATASET = args.dataset_name
# Change this to simulate target dataset
# TARGET_DATASET = "WenyaoDataset"
TARGET_DATASET = "Cholec80_clean"
# TARGET_DATASET = None


def generate_smoke_2d(h, w, scale=100, density=0.25):
    x_seed = np.random.randint(-h, h + 1)
    y_seed = np.random.randint(-w, w + 1)
    cloud = np.zeros((h, w))
    for x in range(h):
        for y in range(w):
            cloud[x][y] = pnoise2((x + x_seed) / scale, (y + y_seed) / scale, octaves=5, persistence=0.5, lacunarity=2.0)
    return (cloud - cloud.min()) / (cloud.max() - cloud.min()) * density  # Normalize to 0-1


def generate_smoke_3d(h, w, t, r_x, r_y, scale=0.1, density=0.25):
    # x_seed = np.random.randint(-h, h + 1)
    # y_seed = np.random.randint(-w, w + 1)
    cloud = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            # Generate 3D Perlin noise (x, y, time)
            cloud[i][j] = pnoise3((j + r_x) / scale, (i + r_y) / scale, t, octaves=5, persistence=0.5, lacunarity=2.0, base=300)
    return (cloud - cloud.min()) / (cloud.max() - cloud.min()) * density  # Normalize to 0-1


class Render:
    def __init__(self, dataset_name):
        self.root_dir = ".."
        self.dataset_name = dataset_name
        self.dataset_path = f"{self.root_dir}/dataset/{dataset_name}"
        self.dataset_gt_path = self.dataset_path + "/GT"

        self.src_img_path_list = glob.glob(self.dataset_gt_path + "/*/*.*")

        self.video_frame_name_list = list(map(lambda x: x.split("/")[-2:], self.src_img_path_list))

        self.video_dict = {x[0]: [] for x in self.video_frame_name_list}

        for video_frame in self.video_frame_name_list:
            video, frame = video_frame

            self.video_dict[video].append(frame)

        for video in self.video_dict:
            self.video_dict[video] = sorted(self.video_dict[video])

        self.video_list = list(sorted(self.video_dict.keys()))

        self.smoke_syn_dir = f"{self.dataset_path}/smoke"
        self.x_syn_dir = f"{self.dataset_path}/X"
        os.makedirs(self.x_syn_dir, exist_ok=True)
        os.makedirs(self.smoke_syn_dir, exist_ok=True)

    def start_render(self):
        zip_n = len(self.src_img_path_list)

        for i in tqdm.tqdm(range(len(self.video_list)), position=0, leave=False, desc="video"):
            video_name = self.video_list[i]
            frame_list = self.video_dict[video_name]
            random_seed = (np.random.randint(-100, 100 + 1), np.random.randint(-100, 100 + 1))
            smoke_density = np.random.uniform(0.25, 0.75)
            # smoke_scale = 0.1
            smoke_scale = np.random.randint(100, 201)

            for j in tqdm.tqdm(range(len(frame_list)), position=1, leave=False, desc="frame"):
                frame_name = frame_list[j]

                gt_path = f"{self.dataset_gt_path}/{video_name}/{frame_name}"
                smoke_syn_path = f"{self.smoke_syn_dir}/{video_name}/{frame_name}"
                x_syn_path = f"{self.x_syn_dir}/{video_name}/{frame_name}"

                os.makedirs(os.path.split(smoke_syn_path)[0], exist_ok=True)
                os.makedirs(os.path.split(x_syn_path)[0], exist_ok=True)

                img = cv2.imread(gt_path)

                h, w, _ = img.shape

                t = j / 25
                # smoke_scale = np.random.randint(100, 201)
                # smoke_density = np.random.uniform(0.25, 0.75)
                smoke_mask = generate_smoke_3d(h, w, t, random_seed[0], random_seed[1], smoke_scale, smoke_density)
                # smoke_mask = generate_smoke_2d(h, w, smoke_scale, smoke_density)

                smoke_mask = smoke_mask.reshape(h, w, 1)

                # if TARGET_DATASET == "Cholec80_clean":
                    # background_mask = img == 0
                    # background_mask = (~np.logical_and(np.logical_and(img[..., 0] < 5, img[..., 1] < 5), img[..., 2] < 5)).astype(np.uint8) * 255

                    # erode = cv2.erode(background_mask, kernel=(3, 3), iterations=5)
                    # dilate = cv2.dilate(erode, kernel=(3, 3), iterations=5)

                    # smoke_mask[dilate == 0, :] = 0

                x_syn = np.clip(img * (1 - smoke_mask) + 255.0 * smoke_mask, 0, 255).astype(np.uint8)

                cv2.imwrite(smoke_syn_path, (smoke_mask[:, :, 0] * 255).astype(np.uint8))

                cv2.imwrite(x_syn_path, x_syn)


def main():

    if not TARGET_DATASET:
        return
    render = Render(TARGET_DATASET)
    render.start_render()


if __name__ == '__main__':
    main()


