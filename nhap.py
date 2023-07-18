import cv2
import os
import numpy as np

image_folder = "/home/hadoop/PycharmProjects/CPS_landmark/sample_images"

gt = []
pred = []
h, w = 256, 256
for name in os.listdir(image_folder):
    image = cv2.imread(os.path.join(image_folder, name))
    image_gt = image[:, :w // 2, :]
    image_pred = image[:, w // 2:, :]
    gt.append(image_gt)
    pred.append(image_pred)

image_gt = np.concatenate(gt, axis=1)
image_pred = np.concatenate(pred, axis=1)

cv2.imwrite("/home/hadoop/Downloads/gt_side_da.png", image_gt)
cv2.imwrite("/home/hadoop/Downloads/pred_side_da.png", image_pred)
