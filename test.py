import random

import torch

from datasets.aflw import AFLW
from models.network import Network
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

parser = ArgumentParser(description="testing")
parser.add_argument("--mode", type=str, default='joint', help='joint or channel')
args = parser.parse_args()
warnings.filterwarnings('ignore')
config = get_config(train=False)
# config.device = 'cpu'
dataset = AFLW(config.test_text,
               config.test_annotations_path,
               config.test_images_path,
               training=False,
               unsupervised=False,
               mean=config.mean,
               std=config.std)

model = Network(num_classes=config.num_classes,
                model_size=config.model_size,
                model=config.model)

model = DataParallel(model).to(config.device)

checkpoint_name = "resnet18_fully_supervised_1_8_0"
checkpoint_path = os.path.join("./log/snapshot", checkpoint_name, 'checkpoint_best.pt')
checkpoint = torch.load(checkpoint_path, map_location=config.device)
model.module.load_state_dict(checkpoint['state_dict'])
model.eval()

nme_meter1 = AverageMeter()
nme_meter2 = AverageMeter()
evaluator = NME(h=config.img_height, w=config.img_width)
save_dir = "./sample_images"
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.mkdir(save_dir)
l = len(dataset)
index = list(range(l))
pbar = tqdm(index, total=len(index), desc=f'Testing {checkpoint_name}')
for i in pbar:
    data = dataset[i]
    image = data['image']
    heatmap = data['heatmap']  # 20 x H x W
    mask = data['mask_heatmap'][0]
    landmark = data['landmark'].astype(np.int)
    # print(landmark.shape)
    # print(heatmap.shape)
    # print(mask.shape)
    image_save = image.copy()
    image = torch.from_numpy(image).unsqueeze(0).to(config.device)
    with torch.no_grad():
        pred1 = model(image, step=1)
        pred1 = torch.sigmoid(pred1)
        pred1_ = pred1.clone()

        pred2 = model(image, step=2)
        pred2 = torch.sigmoid(pred2)
        pred2_ = pred2.clone()
    pred = pred1.squeeze(0).cpu().numpy()  # 20 x H x W
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
    landmark_pred_2 = heatmap2coordinate(pred2_)
    nme1 = evaluator(landmark_pred_1, landmark_gt, torch.from_numpy(mask[:-1]).unsqueeze(0).to(config.device))
    nme2 = evaluator(landmark_pred_2, landmark_gt, torch.from_numpy(mask[:-1]).unsqueeze(0).to(config.device))

    nme_meter1.update(nme1.item())
    nme_meter2.update(nme2.item())
    landmark_pred = landmark_pred_1.squeeze(0).int()
    if i < 100:
        if args.mode == 'channel':
            for p in range(19):
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
            hm_gt = np.concatenate([hm_gt, hm_gt, hm_gt], axis=0).transpose(1, 2, 0) * 255.0
            hm_pred = np.concatenate([hm_pred, hm_pred, hm_pred], axis=0).transpose(1, 2, 0) * 255.0
            hm_gt = hm_gt.astype(np.uint8)
            hm_pred = hm_pred.astype(np.uint8)
            image_save_pred = image_save.copy()
            for p in range(19):
                gt_points = int(landmark[p][0].item()), int(landmark[p][1].item())
                pred_points = int(landmark_pred[p][0].item()), int(landmark_pred[p][1].item())
                if mask[p] == 1:
                    image_save = cv2.circle(image_save, gt_points, 1, (255, 0, 0), 2)  # blue
                    cv2.circle(image_save_pred, pred_points, 1, (0, 0, 255), 2)  # red
            save1 = np.concatenate([image_save, hm_gt], axis=0)
            save2 = np.concatenate([image_save_pred, hm_pred], axis=0)
            save = np.concatenate([save1, save2], axis=1)

            cv2.imwrite(os.path.join(save_dir, f"{i}.png"), save)
    pbar.set_postfix({"NME": [nme_meter1.average(), nme_meter2.average()]})
print(colored(f"{checkpoint_name}: {nme_meter1.average()}, {nme_meter2.average()}", 'red'))