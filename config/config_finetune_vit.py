from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pathlib
import sys
import time
import numpy as np
from easydict import EasyDict as edict
from pathlib import Path
import argparse
from termcolor import colored
import json


def makedir(dir, make_if_exist=False, makedir=True):
    name = dir.split('/')[-1]  # base name
    parent = "/".join(dir.split("/")[:-1])
    exist = False
    sim_dir = []
    for d in os.listdir(parent):
        if name in d:
            exist = True
            if name == d:
                sim_dir.append(d + "_0")
            else:
                sim_dir.append(d)
    if makedir:
        if not exist:  # dir not exists yet
            name = name + "_0"
            os.mkdir(os.path.join(parent, name))
            return os.path.join(parent, name)
        elif not make_if_exist:
            print("directory already exist")
            exit(0)
        else:
            latest_dir = sorted(sim_dir, key=lambda x: int(x.split("_")[-1]))[-1]
            nb = int(latest_dir.split("_")[-1]) + 1
            name = name + f"_{nb}"
            os.mkdir(os.path.join(parent, name))
            return os.path.join(parent, name)
    else:
        if len(sim_dir) == 0:
            return dir + "_0"
        else:
            latest_dir = sorted(sim_dir, key=lambda x: int(x.split("_")[-1]))[-1]
            return os.path.join(parent, latest_dir)


def get_config(train=True):
    C = edict()
    config = C

    '''PATH CONFIG'''
    C.seed = 42
    C.log_dir = "./log_vit"
    if not os.path.exists(C.log_dir):
        pathlib.Path(os.path.join(C.log_dir, 'snapshot')).mkdir(parents=True, exist_ok=True)
    C.train_text_labeled = '/home/s/chaunm/CPS_landmarks/data/split_side/train_labeled_1_1.txt'
    C.train_text_unlabeled = '/home/s/chaunm/CPS_landmarks/data/split_side/train_unlabeled_1_2.txt'
    C.test_text = "/home/s/chaunm/CPS_landmarks/data/split_side/test.txt"
    C.train_annotations_path = '/home/s/chaunm/DATA/SIDE/train_side_paper_128_4/annotations'
    C.train_images_path = '/home/s/chaunm/DATA/SIDE/train_side_paper_128_4/images'
    C.test_annotations_path = '/home/s/chaunm/DATA/SIDE/test_side_paper_128_4/annotations'
    C.test_images_path = '/home/s/chaunm/DATA/SIDE/test_side_paper_128_4/images'

    '''DATASET CONFIG'''
    C.mean = [0.485, 0.456, 0.406]
    C.std = [0.229, 0.224, 0.225]

    '''MODEL CONFIG'''
    C.encoder_embedding_dim = 768
    C.decoder_embedding_dim = 512
    C.encoder_layers = 12
    C.decoder_layers = 4
    C.n_heads_encoder_layer = 12
    C.n_heads_decoder_layer = 16
    C.patch_size = 8
    C.num_classes = 5
    C.freeze_encoder = False
    # C.pretrained_path = "/home/s/chaunm/CPS_landmarks/MAE/log_mae/snapshot/vit-mae_2/checkpoint_best.pt"  # for 19 points
    # C.pretrained_path = "/home/s/chaunm/CPS_landmarks/MAE/log_mae/snapshot/vit-mae_6/checkpoint_best.pt"  # for 5 points
    # C.pretrained_path = "/home/s/chaunm/CPS_landmarks/MAE/log_mae/snapshot/vit-mae_8/checkpoint_best.pt"  # for 4 points

    C.pretrained_path = None
    '''TRAIN CONFIG'''
    C.img_height = 128
    C.img_width = 128
    C.train_batch_size = 16
    C.test_batch_size = 16
    C.num_workers = 8
    C.lr = 1e-3
    C.lr_scheduler = 'cosine'
    C.clip_norm = None  # if freeze encoder = False, reduce lr, warm_up lr, clip norm = 1
    C.weight_decay = 1e-4
    C.warm_up_step = 100
    C.total_epoch = 100
    C.device = 'cuda:0'

    '''LOSS CONFIG'''
    C.loss = 'mse'
    C.alpha = 2.1
    C.omega = 14
    C.epsilon = 1
    C.theta = 0.5
    C.use_target_weight = False
    C.loss_weight = 1

    '''NAME CONFIG'''
    C.name = "vit-unetr-pretrained" if C.pretrained_path is not None else 'vit-unetr-from-scratch'
    C.snapshot_dir = os.path.join(C.log_dir, 'snapshot', C.name)
    C.snapshot_dir = makedir(C.snapshot_dir, make_if_exist=True, makedir=train)
    C.name = C.snapshot_dir.split("/")[-1]
    if train:
        with open(os.path.join(C.snapshot_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
    print(colored(C.snapshot_dir, 'red'))
    return config
