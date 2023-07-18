import random

import cv2
import torch.utils.data
from utils.utils import heatmap2coordinate
from datasets.aflw import AFLW
import numpy as np
import os
import math
import torch.nn.functional as F
import shutil
from scipy.ndimage import grey_dilation
from utils.utils import generate_gaussian, generate_pseudo_label

dataset = AFLW(
    '/home/s/chaunm/CPS_landmarks/data/split_side/train_labeled_1_8.txt',
    '/home/s/chaunm/DATA/SIDE/train_side_paper_128_4/annotations',
    '/home/s/chaunm/DATA/SIDE/train_side_paper_128_4/images',
    unsupervised=False,
    training=True,
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
indices = list(range(len(dataset)))


def soft_argmax(heatmap):
    """
    Apply soft argmax on a given heatmap tensor to compute landmark coordinates.
    Args:
        heatmap: PyTorch tensor of shape B x C x H x W representing the heatmap.
    Returns:
        landmark: PyTorch tensor of shape B x C x 2 representing the landmark coordinates.
    """
    device = heatmap.device
    # print(heatmap.shape)
    batch_size, channels, height, width = heatmap.shape
    softmax = F.softmax(heatmap.view(batch_size, channels, -1) * 128 * 128, dim=-1).view(batch_size, channels, height,
                                                                                         width)

    indices_kernel = torch.arange(start=0, end=height * width).unsqueeze(0).view(height
                                                                                 , width)
    # Create a grid of coordinates
    conv = softmax * indices_kernel
    indices = conv.sum(2).sum(2)  # B x C
    # print(indices)
    # x_coords = torch.linspace(0, 1, width, device=device).view(1, 1, 1, width).expand(batch_size, -1, height, -1)
    # y_coords = torch.linspace(0, 1, height, device=device).view(1, 1, height, 1).expand(batch_size, -1, -1, width)
    #
    # # Compute the expected x and y coordinates
    # expected_x = torch.sum(x_coords * softmax, dim=(2, 3)).unsqueeze(-1)
    # expected_y = torch.sum(y_coords * softmax, dim=(2, 3)).unsqueeze(-1)
    expected_x = indices % width
    expected_y = indices.floor() / height
    # Concatenate the expected x and y coordinates to get the landmark coordinates
    landmark = torch.cat([expected_x.unsqueeze(-1), expected_y.unsqueeze(-1)], dim=2)
    # print(landmark.shape)
    return landmark


import torch
import torch.nn.functional as F


def channelwise_dilation(input_tensor, kernel_size):
    """
    Performs dilation on each channel of a PyTorch tensor of size BxKxHxW.

    Args:
        input_tensor (torch.Tensor): Input tensor with shape BxKxHxW.
        kernel_size (int): Size of the square kernel for dilation.

    Returns:
        torch.Tensor: Output tensor with shape BxKxHxW.
    """
    # Split the input tensor into a list of channel tensors.
    # channel_tensors = torch.split(input_tensor, 1, dim=1)
    #
    # # Apply dilation to each channel tensor.
    # dilated_tensors = []
    # for channel_tensor in channel_tensors:
    #     dilated_channel = F.max_pool2d(channel_tensor, kernel_size, stride=1, padding=kernel_size // 2)
    #     dilated_tensors.append(dilated_channel)

    # Concatenate the dilated channel tensors back into a single tensor.
    output_tensor = F.max_pool2d(input_tensor, kernel_size, stride=1, padding=kernel_size // 2)
    # output_tensor = torch.cat(dilated_tensors, dim=1)

    return output_tensor


gaussian_mask = generate_gaussian()
for i in indices:
    data = dataset[i]
    image = data['image']
    heatmap = data['heatmap']
    # heatmap = channelwise_dilation(torch.from_numpy(heatmap).unsqueeze(0), 3).squeeze(0).numpy()
    landmark = data['landmark']
    mask = data['mask_heatmap']
    heatmap_ = torch.from_numpy(heatmap)

    image = (image.transpose(1, 2, 0) * dataset.std + dataset.mean) * 255.0
    image = image.astype(np.uint8)
    total_save = []

    hm_gt = heatmap[:, :, :] * 255.0
    hm_gt = hm_gt.astype(np.uint8)
    print(mask)
    # if np.sum(mask) == 1:

    for p in range(5):
        if p < 4:
            point = int(landmark[p][0].item()), int(landmark[p][1].item())
            image = cv2.circle(image, point, 1, (255, 0, 0), 1)
        hm_p = np.concatenate([hm_gt[p:p + 1], hm_gt[p:p + 1], hm_gt[p:p + 1]], axis=0).transpose(1, 2, 0)
        # hm_pd_p = np.concatenate([hm_pd[p:p + 1], hm_pd[p:p + 1], hm_pd[p:p + 1]], axis=0).transpose(1, 2, 0)
        save = np.concatenate([image.copy(), hm_p], axis=0)
        total_save.append(save)
    total_save = np.concatenate(total_save, axis=1)
    cv2.imwrite(os.path.join(save_dir, f"{i}.png"),
                total_save)
# print(headpose * 180 / math.pi)
