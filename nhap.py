import math
import os
import random
import shutil

import torch
from scipy.io import loadmat
import numpy as np
import cv2

mat_dir = "/home/s/chaunm/DATA/AFLW/train/annotations"
image_path = '/home/s/chaunm/DATA/AFLW/train/images/'
save_dir = "./sample_images"
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.mkdir(save_dir)
mat_list = random.choices(os.listdir(mat_dir), k=10)
for file in mat_list:
    file_path = os.path.join(mat_dir, file)
    annotation = loadmat(file_path)
    image_name = annotation['image_name']
    # print(type(image_name[0]))

    image = cv2.imread(os.path.join(image_path, str(image_name[0])))

    landmark = annotation['landmark']
    heatmap = annotation['heatmap'].transpose(2, 0, 1)
    headpose = annotation['headpose']
    mask_heatmap = annotation['mask_heatmap']
    heatmap_ = torch.from_numpy(heatmap).reshape(20, -1)[:-1]  # ignore background channel
    _, index = torch.max(heatmap_, dim=-1, keepdim=True)
    xs = index % 128
    ys = index // 128
    lms = torch.concat([xs, ys], dim=-1)
    print(lms)
    print(landmark)
    break
    # print(heatmap.shape)
    # hm_save = np.sum(heatmap[:, :, :-1], axis=2, keepdims=True)
    # hm_save = np.concatenate([hm_save, hm_save, hm_save], axis=2)
    # hm_save = (hm_save * 255.0).astype(np.uint8)
    # for p in range(len(landmark)):
    #     point = (int(landmark[p][0]), int(landmark[p][1]))
    #     image = cv2.circle(image, point, 1, (255, 0, 0), 2)
    # cv2.imwrite(os.path.join(save_dir, str(image_name[0])),
    #             np.concatenate([image, hm_save], axis=0))
    # print(headpose * 180 / math.pi)
