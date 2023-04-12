import random

import cv2
import torch.utils.data
from utils.utils import heatmap2coordinate
from datasets.aflw import AFLW
import numpy as np
import os
import math
import shutil

dataset = AFLW(
    '/home/s/chaunm/CPS_landmarks/data/split/train_labeled_1_8.txt',
    '/home/s/chaunm/DATA/AFLW/train_128_4/annotations',
    '/home/s/chaunm/DATA/AFLW/train_128_4/images',
    unsupervised=False,
    training=True
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
# d = next(iter(dataloader))
# heatmap = d['heatmap']
# mask = d['mask_heatmap'][:, 0, :-1].unsqueeze(2)
# landmarks = d['landmark'].int()
# # print(heatmap.shape)
# # print(landmarks.shape)
# heatmap = heatmap.view(8, 20, -1)
# _, index = torch.max(heatmap, dim=-1, keepdim=True)
# index = index[:, :-1, :]
# xs = index % 128
# ys = index // 128
# lms = torch.cat([xs, ys], dim=-1)
# # print(landmarks.shape, mask.shape, lms.shape)
# # print(torch.sum(lms - landmarks * mask))
# diff = lms - landmarks * mask
# mask2 = torch.where(diff > 1, 1, 0)
# print(torch.sum(diff * mask2))
# # print(lms[0])
# # print((landmarks * mask)[0])
save_dir = "./sample_images"
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.mkdir(save_dir)
indices = random.choices(list(range(len(dataset))), k=100)
for i in indices:
    data = dataset[i]
    image = data['image']
    heatmap = data['heatmap']
    landmark = heatmap2coordinate(torch.from_numpy(heatmap).unsqueeze(0).contiguous()).squeeze(0).cpu().numpy()
    image = (image.transpose(1, 2, 0) * dataset.std + dataset.mean) * 255.0
    image = image.astype(np.uint8)
    total_save = []
    # hm_save = np.sum(heatmap[:-1, :, :], axis=0, keepdims=True).transpose(1, 2, 0)
    for p in range(19):
        hm_gt = heatmap[p, :, :]
        hm_gt = np.expand_dims(hm_gt, 0)
        print(np.min(hm_gt), np.max(hm_gt), np.sum(hm_gt))
        hm_gt = np.concatenate([hm_gt, hm_gt, hm_gt], axis=0).transpose(1, 2, 0) * 255.0
        hm_gt = hm_gt.astype(np.uint8)
        image_save_ = image.copy()
        point = (int(landmark[p][0]), int(landmark[p][1]))
        image_save_ = cv2.circle(image_save_, point, 1, (255, 0, 0), 2)
        save = np.concatenate([image_save_, hm_gt.copy()], axis=0)
        total_save.append(save)
    total_save = np.concatenate(total_save, axis=1)
    cv2.imwrite(os.path.join(save_dir, f"{i}.png"),
                total_save)
    # print(headpose * 180 / math.pi)
