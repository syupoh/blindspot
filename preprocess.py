import os
import sys
import glob
import argparse

import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tqdm import tqdm
from utils import get_file_paths, LABELS

def argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--source', '-s', type=str, help='souce directory path to preprocess')
    parser.add_argument('--target', "-t", type=str, help='save directory path')
    parser.add_argument('--cpu_count', type=int, help='cpu count to do multi processing', default=16)
    return parser.parse_args()


def get_x_position(pixel_array, th, reverse):
    min_value = 0xffff
    for row in range(pixel_array.shape[0])[::reverse]:
        avg_value = int(pixel_array[row, :, 0].mean())
        min_value = min(avg_value, min_value)
        if avg_value >= min_value + th:
            return row
    return -1


def get_y1_position(pixel_array, th):
    for col in reversed(range(int(pixel_array.shape[1]/2))):
        avg_value = int(pixel_array[:, col, 0].mean())
        next_avg_value = int(pixel_array[:, col-1, 0].mean())
        if abs(avg_value - next_avg_value) >= th:
            return col
    return -1


def get_y2_position(pixel_array, th):
    for col in reversed(range(pixel_array.shape[1])):
        avg_value = int(pixel_array[:, col, :].mean())
        if avg_value >= th:
            return col
    return -1


def preprocess(path):
    dirname = os.path.basename(os.path.dirname(path))
    fname = os.path.basename(path)
    target_dir_path = os.path.join(args.target, dirname)
    save_path = os.path.join(target_dir_path, fname)
    # print(target_dir_path)
    # print(save_path)

    # try:
    #     if not os.path.exists(target_dir_path):
    #         os.mkdir(target_dir_path)
    #         print(target_dir_path)
    # except:
    #     ...

    if not os.path.exists(save_path):
        try:
            img = mpimg.imread(path)
            x1 = get_x_position(img, th=10, reverse=1)
            x2 = get_x_position(img, th=30, reverse=-1)
            y1 = get_y1_position(img, th=15)
            y2 = get_y2_position(img, th=30)
            img = img[x1:x2+1, y1:y2+1, :]
            plt.imsave(save_path, img, format="jpg")
        except Exception as e:
            print(e, path)


if __name__ == "__main__":
    args = argparser()
    file_paths = get_file_paths(args.source)
    if not os.path.exists(args.target):
        os.mkdir(args.target)
    for site in LABELS:
        if not os.path.exists(os.path.join(args.target, site)):
            os.mkdir(os.path.join(args.target, site))

    with mp.Pool(args.cpu_count) as pool:
        for _ in tqdm(pool.imap_unordered(preprocess, file_paths), total=len(file_paths), desc="preprocessing"):
            ...