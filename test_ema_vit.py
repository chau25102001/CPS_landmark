from datasets.aflw import AFLW
from models.mean_teacher_network_vit import MeanTeacher_CPS_ViT
from config.config_ema_vit import get_config
import os
import numpy as np
import math
import cv2
import shutil
import torch
from utils.utils import heatmap2coordinate, NME, AverageMeter
import warnings
from argparse import ArgumentParser
from tqdm import tqdm
from termcolor import colored
import matplotlib.pyplot as plt
from utils.utils import merge_dict

parser = ArgumentParser(description="testing")
parser.add_argument("--mode", type=str, default='joint', help='joint or channel')
parser.add_argument("--checkpoint_path", type=str, help="path to checkpoint .pt file")
parser.add_argument("--test_text", type=str, help="path to test text file, containing test annotation file names")
parser.add_argument("--test_annotation_path", type=str, help="path to the test annotation folder")
parser.add_argument("--test_images_path", type=str, help="path to the test images")
parser.add_argument("--num_classes", type=int,
                    help="number of keypoint + 1, either 20 for AFLW-19, 6 for AFLW-DA, or 5 for SideFace-DA")
args = parser.parse_args()
warnings.filterwarnings('ignore')
config = get_config(train=False)
config = merge_dict(config, args)
config.pretrained_path = None  # dont need to load pretrained

dataset = AFLW(config.test_text,
               config.test_annotations_path,
               config.test_images_path,
               training=False,
               unsupervised=False,
               mean=config.mean,
               std=config.std)

model = MeanTeacher_CPS_ViT(config)
# checkpoint_name = "vit_ema_cps_pretrained_17"
# checkpoint_path = os.path.join("./log_ema_vit/snapshot", checkpoint_name, 'checkpoint_best.pt')
checkpoint = torch.load(args.checkpoint_path, map_location=config.device)
model.load_state_dict(checkpoint['state_dict'])
model = model.to(config.device)
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
pbar = tqdm(index, total=len(index))
error_total = torch.zeros((1, config.num_classes - 1), device=config.device)
cmap = plt.get_cmap("coolwarm_r")
sm = plt.cm.ScalarMappable(cmap=cmap)
color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
color_range = color_range.reshape(256, 1, 3)
for i in pbar:
    data = dataset[i]
    image = data['image']
    heatmap = data['heatmap']  # 20 x H x W
    mask = data['mask_heatmap']
    landmark = data['landmark'].astype(int)
    # print(landmark.shape)
    # print(heatmap.shape)
    # print(mask.shape)
    image_save = image.copy()
    image = torch.from_numpy(image).unsqueeze(0).to(config.device)
    with torch.no_grad():
        pred1, t_pred1 = model(image, step=1)
        pred1_softmax = pred1.clone().squeeze(0).cpu().numpy()
        pred1 = torch.sigmoid(pred1)
        t_pred1 = torch.sigmoid(t_pred1)

        pred2, t_pred2 = model(image, step=2)
        pred2 = torch.sigmoid(pred2)
        t_pred2 = torch.sigmoid(t_pred2)
    pred = pred1.squeeze(0).cpu().numpy()  # 20 x H x W
    _, h, w = pred.shape
    image_save = (image_save.transpose(1, 2, 0) * dataset.std + dataset.mean) * 255.0
    image_save = np.ascontiguousarray(image_save, dtype=np.uint8)
    save_total = []
    landmark_gt = torch.from_numpy(landmark).unsqueeze(0).to(config.device)
    landmark_pred_1 = heatmap2coordinate(pred1)  # 1 x 19 x 2
    landmark_pred_2 = heatmap2coordinate(pred2)
    landmark_t_pred_1 = heatmap2coordinate(t_pred1)
    landmark_t_pred_2 = heatmap2coordinate(t_pred2)
    mask = torch.from_numpy(mask).unsqueeze(-1).to(config.device)

    nme1 = evaluator(landmark_pred_1, landmark_gt, mask)
    nme2 = evaluator(landmark_pred_2, landmark_gt, mask)
    nme_t1 = evaluator(landmark_t_pred_1, landmark_gt, mask)
    nme_t2 = evaluator(landmark_t_pred_2, landmark_gt, mask)
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
            hm_gt = np.max(heatmap[:-1, :, :], axis=0, keepdims=False)
            hm_pred = np.max(pred[:-1, :, :], axis=0, keepdims=False)  # 1, h, w -> index ()
            # hm_gt = np.concatenate([hm_gt, hm_gt, hm_gt], axis=0).transpose(1, 2, 0) * 255.0
            # hm_pred = np.concatenate([hm_pred, hm_pred, hm_pred], axis=0).transpose(1, 2, 0) * 255.0
            hm_gt = hm_gt * 255.0
            hm_pred = hm_pred * 255.0
            hm_gt = hm_gt.astype(np.uint8)
            hm_pred = hm_pred.astype(np.uint8)
            image_save_pred = image_save.copy()
            image_hm = image_save.copy()
            for p in range(config.num_classes - 1):
                gt_points = int(landmark[p][0].item()), int(landmark[p][1].item())
                pred_points = int(landmark_pred[p][0].item()), int(landmark_pred[p][1].item())
                # if mask[p] == 1:
                image_save = cv2.circle(image_save, gt_points, 1, (255, 0, 0), 2)  # blue
                image_save_pred = cv2.circle(image_save_pred, pred_points, 1, (0, 0, 255), 2)  # red
            hm_gt = cv2.applyColorMap(hm_gt, color_range)
            hm_pred = cv2.applyColorMap(hm_pred, color_range)

            hm_gt = (hm_gt * 0.4 + image_hm * 0.6).astype(np.uint8)
            hm_pred = (hm_pred * 0.4 + image_hm * 0.6).astype(np.uint8)

            save1 = np.concatenate([image_save, hm_gt], axis=0)
            save2 = np.concatenate([image_save_pred, hm_pred], axis=0)
            # save2 = cv2.resize(save2, (256, 512), cv2.INTER_LINEAR)
            save = np.concatenate([save1, save2], axis=1)

            cv2.imwrite(os.path.join(save_dir, f"{i}.png"), save)
    pbar.set_postfix({"NME": [nme_meter1.average(), nme_meter2.average()]})
print(colored(
    f"NME: {nme_meter1.average()}, {nme_meter2.average()}, {nme_meter_t1.average()}, {nme_meter_t2.average()}",
    'red'))
print(error_total / l)
