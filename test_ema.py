import math
import random

import torch

from datasets.aflw import AFLW
from models.mean_teacher_network import MeanTeacher_CPS
from config.config import get_config
from torch.nn import DataParallel
import os
import numpy as np
import cv2
import shutil
from utils.utils import heatmap2coordinate, NME, AverageMeter
import warnings
from argparse import ArgumentParser
from tqdm import tqdm
from termcolor import colored
import torch.nn.functional as F

parser = ArgumentParser(description="testing")
parser.add_argument("--mode", type=str, default='joint', help='joint or channel')
args = parser.parse_args()
warnings.filterwarnings('ignore')
config = get_config(train=False)


# config.device = 'cpu'
# print(config.device)

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


dataset = AFLW(config.test_text,
               config.test_annotations_path,
               config.test_images_path,
               training=False,
               unsupervised=False,
               mean=config.mean,
               std=config.std)

model = MeanTeacher_CPS(num_classes=config.num_classes,
                        model_size=config.model_size,
                        model=config.model)

model = DataParallel(model).to(config.device)

checkpoint_name = "hgnet18_CPS_EMA1_2_5/"
checkpoint_path = os.path.join("./log/snapshot", checkpoint_name, 'checkpoint_best.pt')
checkpoint = torch.load(checkpoint_path, map_location=config.device)
model.module.load_state_dict(checkpoint['state_dict'])
model.eval()

nme_meter1 = AverageMeter()
nme_meter2 = AverageMeter()
nme_meter_t1 = AverageMeter()
nme_meter_t2 = AverageMeter()
evaluator = NME(h=config.img_height, w=config.img_width)
save_dir = "./sample_images"
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.mkdir(save_dir)
l = len(dataset)
index = list(range(l))
pbar = tqdm(index, total=len(index), desc=f'Testing {checkpoint_name}')
error_total = torch.zeros((1, config.num_classes - 1), device=config.device)
for i in pbar:
    data = dataset[i]
    image = data['image']
    heatmap = data['heatmap']  # 20 x H x W
    mask = data['mask_heatmap'][0]
    landmark = data['landmark'].astype(int)
    # print(landmark.shape)
    # print(heatmap.shape)
    # print(mask.shape)
    image_save = image.copy()
    image = torch.from_numpy(image).unsqueeze(0).to(config.device)
    with torch.no_grad():
        pred1, t_pred1, _, _ = model(image, step=1)
        pred1_softmax = pred1.clone().squeeze(0).cpu().numpy()
        pred1 = torch.sigmoid(pred1)
        t_pred1 = torch.sigmoid(t_pred1)

        pred2, t_pred2, _, _ = model(image, step=2)
        pred2 = torch.sigmoid(pred2)
        t_pred2 = torch.sigmoid(t_pred2)

    pred = pred1.squeeze(0).cpu().numpy()  # 20 x H x W
    _, h, w = pred.shape
    # hm_gt = np.max(heatmap[:-1, :, :], axis=0)  # 1 x H x W
    # hm_pred = np.max(pred[:-1, :, :], axis=0)
    # hm_gt = np.expand_dims(hm_gt, 0)
    # hm_pred = np.expand_dims(hm_pred, 0)
    image_save = (image_save.transpose(1, 2, 0) * dataset.std + dataset.mean) * 255.0
    image_save = np.ascontiguousarray(image_save, dtype=np.uint8)
    save_total = []
    landmark_gt = torch.from_numpy(landmark).unsqueeze(0).to(config.device)
    # landmark_gt = heatmap2coordinate(
    #     torch.from_numpy(heatmap).unsqueeze(0).contiguous().to(config.device))
    # landmark_gt = landmark_gt.squeeze(0).int()

    landmark_pred_1 = heatmap2coordinate(pred1)  # 1 x 19 x 2
    landmark_pred_2 = heatmap2coordinate(pred2)
    landmark_t_pred_1 = heatmap2coordinate(t_pred1)
    landmark_t_pred_2 = heatmap2coordinate(t_pred2)

    nme1 = evaluator(landmark_pred_1, landmark_gt, None)
    nme2 = evaluator(landmark_pred_2, landmark_gt, None)
    nme_t1 = evaluator(landmark_t_pred_1, landmark_gt, None)
    nme_t2 = evaluator(landmark_t_pred_2, landmark_gt, None)
    error = torch.sqrt(torch.sum((landmark_t_pred_1 - landmark_gt) ** 2, dim=-1)) / math.sqrt(
        config.img_height * config.img_width) * 100  # 1, 19
    error_total += error

    nme_meter1.update(nme1.item())
    nme_meter2.update(nme2.item())

    nme_meter_t1.update(nme_t1.item())
    nme_meter_t2.update(nme_t2.item())
    landmark_pred = landmark_pred_1.squeeze(0).int()
    if i < 100:
        if args.mode == 'channel':
            for p in range(config.num_classes - 1):
                hm_gt = heatmap[p, :, :]
                # hm_pred = torch.softmax(torch.from_numpy(pred1_softmax[p:p + 1, :, :]).view(1, -1), dim=-1).view(h, w) * 10
                hm_pred = pred[p, :, :]
                # print(hm_pred.unique())
                # hm_pred = torch.clamp(hm_pred, min=0, max=1).numpy()
                hm_gt = np.expand_dims(hm_gt, 0)
                hm_pred = np.expand_dims(hm_pred, 0)
                hm_gt = np.concatenate([hm_gt, hm_gt, hm_gt], axis=0).transpose(1, 2, 0) * 255.0
                hm_pred = np.concatenate([hm_pred, hm_pred, hm_pred], axis=0).transpose(1, 2, 0) * 255.0
                hm_gt = hm_gt.astype(np.uint8)
                hm_pred = hm_pred.astype(np.uint8)

                image_save_ = image_save.copy()
                gt_points = landmark[p][0].item(), landmark[p][1].item()
                pred_points = landmark_pred[p][0].item(), landmark_pred[p][1].item()
                # print(image_save.shape, gt_points, pred_points)
                if mask[p] == 1:  # draw only visible point
                    image_save_ = cv2.circle(image_save_, gt_points, 1, (255, 0, 0), 1)  # blue
                    image_save_ = cv2.circle(image_save_, pred_points, 1, (0, 0, 255), 1)  # red
                save = np.concatenate([image_save_, hm_gt.copy(), hm_pred.copy()], axis=0)
                save_total.append(save)
            save_total = np.concatenate(save_total, axis=1)
            cv2.imwrite(os.path.join(save_dir, f"{i}.png"), save_total)
        elif args.mode == 'joint':
            hm_gt = np.max(heatmap[:-1, :, :], axis=0, keepdims=True)
            hm_pred = np.max(pred[:-1, :, :], axis=0, keepdims=True)  # 1, h, w -> index ()
            hm_gt = np.concatenate([hm_gt, hm_gt, hm_gt], axis=0).transpose(1, 2, 0) * 255.0
            hm_pred = np.concatenate([hm_pred, hm_pred, hm_pred], axis=0).transpose(1, 2, 0) * 255.0
            hm_gt = hm_gt.astype(np.uint8)
            hm_pred = hm_pred.astype(np.uint8)
            image_save_pred = image_save.copy()
            for p in range(config.num_classes - 1):
                gt_points = int(landmark[p][0].item()), int(landmark[p][1].item())
                pred_points = int(landmark_pred[p][0].item()), int(landmark_pred[p][1].item())
                # if mask[p] == 1:
                image_save = cv2.circle(image_save, gt_points, 1, (255, 0, 0), 2)  # blue
                cv2.circle(image_save_pred, pred_points, 1, (0, 0, 255), 2)  # red
            save1 = np.concatenate([image_save, hm_gt], axis=0)
            save2 = np.concatenate([image_save_pred, hm_pred], axis=0)
            save = np.concatenate([save1, save2], axis=1)

            cv2.imwrite(os.path.join(save_dir, f"{i}.png"), save)
    pbar.set_postfix({"NME": [nme_meter1.average(), nme_meter2.average()]})
print(colored(
    f"{checkpoint_name}: {nme_meter1.average()}, {nme_meter2.average()}, {nme_meter_t1.average()}, {nme_meter_t2.average()}",
    'red'))
print(error_total / l)
