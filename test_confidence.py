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
from utils.utils import heatmap2coordinate, NME, AverageMeter, local_filter
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

dataset = AFLW(config.train_text_unlabeled,
               config.train_annotations_path,
               config.train_images_path,
               training=False,
               unsupervised=False,
               mean=config.mean,
               std=config.std)

model = MeanTeacher_CPS(num_classes=config.num_classes,
                        model_size=config.model_size,
                        model=config.model)

model = DataParallel(model).to(config.device)
checkpoint_name = "hgnet18_CPS_EMA1_8_0/"
checkpoint_path = os.path.join("./log/snapshot", checkpoint_name, 'checkpoint_best.pt')
checkpoint = torch.load(checkpoint_path, map_location=config.device)
model.module.load_state_dict(checkpoint['state_dict'])
model.eval()
l = len(dataset)
index = list(range(l))
pbar = tqdm(index, total=len(index), desc=f'Testing {checkpoint_name}')
logfile = open("./confidence_1_8.txt", 'w')
for i in pbar:
    data = dataset[i]
    name = data['name']
    image = data['image']
    heatmap = data['heatmap']  # 20 x H x W
    mask = data['mask_heatmap'][0]
    landmark = data['landmark'].astype(int)
    image = torch.from_numpy(image).unsqueeze(0).to(config.device)

    with torch.no_grad():
        _, t_pred1, _, _ = model(image, step=1)
        t_pred1 = torch.sigmoid(t_pred1)

        _, t_pred2, _, _ = model(image, step=2)
        t_pred2 = torch.sigmoid(t_pred2)

        _, max_t1 = local_filter(t_pred1, config.threshold)
        _, max_t2 = local_filter(t_pred2, config.threshold)
        logfile.write(name + "\n")
        logfile.write(str(max_t1.cpu().numpy().tolist()))
        logfile.write("\n")
        logfile.write(str(max_t2.cpu().numpy().tolist()))
        logfile.write("\n\n")

logfile.close()
