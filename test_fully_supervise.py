import random

import torch

from datasets.aflw import AFLW
from models.network import Network
from models.hgnet import *
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
from utils.utils import generate_gaussian, generate_batch_pseudo_label, merge_dict
import math

parser = ArgumentParser(description="testing")
parser.add_argument("--mode", type=str, default='joint', help='joint or channel')
parser.add_argument("--checkpoint_path", type=str, help="path to checkpoint .pt file")
parser.add_argument("--test_text", type=str, help="path to test text file, containing test annotation file names")
parser.add_argument("--test_annotation_path", type=str, help="path to the test annotation folder")
parser.add_argument("--test_images_path", type=str, help="path to the test images")
parser.add_argument("--num_classes", type=int, help="number of keypoint + 1, either 20 for AFLW-19, 6 for AFLW-DA, or 5 for SideFace-DA")

args = parser.parse_args()
warnings.filterwarnings('ignore')
config = get_config(train=False)
config = merge_dict(config, args)
# config.device = 'cpu'
dataset = AFLW(config.test_text,
               config.test_annotations_path,
               config.test_images_path,
               training=False,
               unsupervised=False,
               mean=config.mean,
               std=config.std)

model = HGNet(num_classes=config.num_classes,
              )

model = DataParallel(model).to(config.device)

# checkpoint_name = "vit-unetr-from-scratch_15"
# checkpoint_path = os.path.join("./log/snapshot", checkpoint_name, 'checkpoint_best.pt')
checkpoint = torch.load(args.checkpoint_path, map_location=config.device)
model.module.load_state_dict(checkpoint['state_dict'])
model.eval()

nme_meter1 = AverageMeter()
evaluator = NME(h=config.img_height, w=config.img_width)
save_dir = "./sample_images"
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.mkdir(save_dir)
l = len(dataset)
index = list(range(l))
pbar = tqdm(index, total=len(index), desc=f'Testing')
error_total = torch.zeros((1, config.num_classes - 1), device=config.device)

gaussian_mask = generate_gaussian()
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
        pred1, _ = model(image)
        pred1 = torch.sigmoid(pred1)
        pseudo_label = generate_batch_pseudo_label(pred1, gaussian_mask)
        pred1_ = pred1.clone()

    pred = pred1.squeeze(0).cpu().numpy()  # 20 x H x W
    pseudo_label = pseudo_label.squeeze(0).cpu().numpy()
    # hm_gt = np.max(heatmap[:-1, :, :], axis=0)  # 1 x H x W
    # hm_pred = np.max(pred[:-1, :, :], axis=0)
    # hm_gt = np.expand_dims(hm_gt, 0)
    # hm_pred = np.expand_dims(hm_pred, 0)
    image_save = (image_save.transpose(1, 2, 0) * dataset.std + dataset.mean) * 255.0
    image_save = np.ascontiguousarray(image_save, dtype=np.uint8)
    save_total = []
    # landmark_gt = heatmap2coordinate(
    #     torch.from_numpy(heatmap).unsqueeze(0).contiguous().to(config.device))  # 1 x 19 x 2
    # landmark_gt = landmark_gt.squeeze(0).int()
    landmark_gt = torch.from_numpy(landmark).unsqueeze(0).to(config.device)

    landmark_pred_1 = heatmap2coordinate(pred1_)  # 1 x 19 x 2
    nme1 = evaluator(landmark_pred_1, landmark_gt, None)
    error = torch.sqrt(torch.sum((landmark_pred_1 - landmark_gt) ** 2, dim=-1)) / math.sqrt(
        config.img_height * config.img_width) * 100  # 1, 19
    error_total += error

    nme_meter1.update(nme1.item())
    landmark_pred = landmark_pred_1.squeeze(0).int()
    if i < 100:
        if args.mode == 'channel':
            for p in range(config.num_classes - 1):
                hm_gt = heatmap[p, :, :]
                hm_pred = pred[p, :, :]
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
                    image_save_ = cv2.circle(image_save_, gt_points, 1, (255, 0, 0), 2)  # blue
                    image_save_ = cv2.circle(image_save_, pred_points, 1, (0, 0, 255), 2)  # red
                save = np.concatenate([image_save_, hm_gt.copy(), hm_pred.copy()], axis=0)
                save_total.append(save)
            save_total = np.concatenate(save_total, axis=1)
            cv2.imwrite(os.path.join(save_dir, f"{i}.png"), save_total)
        elif args.mode == 'joint':
            hm_gt = np.max(heatmap[:-1, :, :], axis=0, keepdims=True)
            hm_pred = np.max(pred[:-1, :, :], axis=0, keepdims=True)
            pseudo_label_ = np.max(pseudo_label[:-1, :, :], axis=0, keepdims=True)
            hm_gt = np.concatenate([hm_gt, hm_gt, hm_gt], axis=0).transpose(1, 2, 0) * 255.0
            hm_pred = np.concatenate([hm_pred, hm_pred, hm_pred], axis=0).transpose(1, 2, 0) * 255.0
            pseudo_label_ = np.concatenate([pseudo_label_, pseudo_label_, pseudo_label_], axis=0).transpose(1, 2,
                                                                                                            0) * 255.0
            hm_gt = hm_gt.astype(np.uint8)
            hm_pred = hm_pred.astype(np.uint8)
            pseudo_label_ = pseudo_label_.astype(np.uint8)
            image_save_pred = image_save.copy()
            for p in range(config.num_classes - 1):
                gt_points = int(landmark[p][0].item()), int(landmark[p][1].item())
                pred_points = int(landmark_pred[p][0].item()), int(landmark_pred[p][1].item())
                if mask[p] == 1:
                    image_save = cv2.circle(image_save, gt_points, 1, (255, 0, 0), 2)  # blue
                    cv2.circle(image_save_pred, pred_points, 1, (0, 0, 255), 2)  # red
            save1 = np.concatenate([image_save, hm_gt], axis=0)
            save2 = np.concatenate([image_save_pred, hm_pred], axis=0)
            save3 = np.concatenate([image_save_pred, pseudo_label_], axis=0)
            save = np.concatenate([save1, save2, save3], axis=1)

            cv2.imwrite(os.path.join(save_dir, f"{i}.png"), save)
    pbar.set_postfix({"NME": nme_meter1.average()})
print(colored(f"NME: {nme_meter1.average()}", 'red'))
print(error_total / l)
