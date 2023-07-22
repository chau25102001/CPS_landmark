# import cv2
# import os
# import numpy as np
#
# image_folder = "/home/hadoop/PycharmProjects/CPS_landmark/sample_images"
#
# gt = []
# pred = []
# h, w = 256, 256
# for name in os.listdir(image_folder):
#     image = cv2.imread(os.path.join(image_folder, name))
#     image_gt = image[:, :w // 2, :]
#     image_pred = image[:, w // 2:, :]
#     gt.append(image_gt)
#     pred.append(image_pred)
#
# image_gt = np.concatenate(gt, axis=1)
# image_pred = np.concatenate(pred, axis=1)
#
# cv2.imwrite("/home/hadoop/Downloads/gt_side_da.png", image_gt)
# cv2.imwrite("/home/hadoop/Downloads/pred_side_da.png", image_pred)
from argparse import ArgumentParser
from tqdm import tqdm
from termcolor import colored
import torch.nn.functional as F

parser = ArgumentParser(description="testing")
parser.add_argument("--mode", type=str, default='joint', help='joint or channel')
parser.add_argument("--checkpoint_path", type=str, help="path to checkpoint .pt file")
parser.add_argument("--test_text", type=str, help="path to test text file, containing test annotation file names")
parser.add_argument("--test_annotation_path", type=str, help="path to the test annotation folder")
parser.add_argument("--test_images_path", type=str, help="path to the test images")
args = parser.parse_args()
if __name__ == "__main__":
    print(vars(args))
