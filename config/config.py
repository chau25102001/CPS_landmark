from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
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

    C.seed = 42
    C.log_dir = "./log"
    C.mean_teacher = False
    C.classification = False
    C.fully_supervised = True
    if C.fully_supervised and C.mean_teacher:
        print("only one mode is allowed, either fully_supervised or mean_teacher, switching to mean teacher by default")
        C.mean_teacher = True
        C.fully_supervised = False
    elif not C.fully_supervised and not C.mean_teacher:
        print(
            "must specify at least one mode, either fully_supervised or mean_teacher, switching to mean teacher by default")
        C.mean_teacher = True
        C.fully_supervised = False

    '''PATH CONFIG'''
    C.train_text_labeled = './data/split_side/train_labeled_1_1.txt'
    C.train_text_unlabeled = './data/split_side/train_unlabeled_1_2.txt'
    C.test_text = './data/split_side/test.txt'
    C.train_annotations_path = '/home/s/chaunm/DATA/SIDE/train_side_paper_128_4/annotations'
    C.train_images_path = '/home/s/chaunm/DATA/SIDE/train_side_paper_128_4/images'
    C.test_annotations_path = '/home/s/chaunm/DATA/SIDE/test_side_paper_128_4/annotations'
    C.test_images_path = '/home/s/chaunm/DATA/SIDE/test_side_paper_128_4/images'
    C.ratio = C.train_text_labeled.split("/")[-1].replace('.txt', '').replace('train_labeled_', '')
    '''DATASET CONFIG'''
    C.mean = [0.485, 0.456, 0.406]
    C.std = [0.229, 0.224, 0.225]
    C.mask_distance = False
    '''MODEL CONFIG'''

    C.model = 'hgnet'
    C.model_size = '18'
    C.num_classes = 5  # 20 classes including background

    '''LOSS CONFIG'''
    C.loss = 'mse'
    C.alpha = 2.1
    C.omega = 14
    C.epsilon = 1
    C.theta = 0.5
    C.use_target_weight = False
    C.loss_weight = 1
    C.use_aux_loss = False
    C.cps_loss_weight = 1.5  # important
    C.class_loss_weight = 1.
    C.use_weighted_mask = False  # for a wing loss

    # assert (C.adaptive_batched and C.threshold_reduction == "none") or (
    #         not C.adaptive_batched and C.threshold_reduction != "none")  # none only work on batched mode
    '''TRAIN CONFIG'''
    C.img_height = 128
    C.img_width = 128
    C.radius = 12  # for generating pseudo-label
    C.train_batch_size = 16
    C.test_batch_size = 16
    C.num_workers = 8
    C.labeled_epoch = 0
    C.joint_epoch = 60  # 45 for 1/2
    C.lr = 1e-3
    C.weight_decay = 1e-3
    C.momentum = 0.99
    C.device = 'cuda:0'
    C.warm_up_epoch = 0
    C.clip_norm = None
    C.adaptive_filter = False
    C.apply_function = True
    # C.adaptive_batched = True
    C.threshold_reduction = "min"  # mean, min and none
    if C.threshold_reduction == "none":
        C.adaptive_batched = True
    else:
        C.adaptive_batched = False
    # C.device = 'cpu'
    '''EMA config'''
    C.threshold = 0.7
    C.class_threshold = 0.7
    C.adaptive_threshold = 1 / C.img_width / C.img_height  # change this
    C.start_ema_decay = 0.5
    C.end_ema_decay = 0.99
    C.ema_linear_epoch = 15  # 5 for 1/8 and 1/4, 10 for 1/2
    C.ema_decay_threshold = 0.9

    if C.fully_supervised:
        C.mode = 'fully_supervised_' + C.ratio
    elif C.mean_teacher:
        C.mode = "CPS_EMA" + C.ratio
    else:
        C.mode = "CPS_" + C.ratio

    if C.classification:
        C.mode += "_classification"
    C.name = C.model + C.model_size + "_" + C.mode

    C.snapshot_dir = os.path.join(C.log_dir, 'snapshot', C.name)
    C.snapshot_dir = makedir(C.snapshot_dir, makedir=train, make_if_exist=True)
    C.name = C.snapshot_dir.split("/")[-1]
    if train:
        with open(os.path.join(C.snapshot_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
    print(colored(C.snapshot_dir, 'red'))
    return config
