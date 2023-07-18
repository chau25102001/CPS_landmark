import os.path
import shutil

import tqdm
from scipy.io import loadmat
import numpy as np
import cv2

all_annotation_file = "/home/s/chaunm/DATA/AFLW/AFLWinfo_release.mat"
image_root = "/home/s/chaunm/DATA/AFLW"
all_annot = loadmat(all_annotation_file)
bboxes = all_annot['bbox']
indices = all_annot['ra'][0] - 1
names = np.array([n[0][0] for n in all_annot['nameList']])
save_dir = "/home/s/chaunm/DATA/AFLW/MAE_images_neck"
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.mkdir(save_dir)
train_indices = indices[:20000]
count = 0
chosen_images = open("/home/s/chaunm/CPS_landmarks/data/mae_image_neck.txt", 'r').readlines()
chosen_images = [n.strip() for n in chosen_images]
for i in tqdm.tqdm(range(len(train_indices)), total=len(range(len(train_indices)))):
    id = train_indices[i]
    name = names[id]
    # print(name)
    if name not in chosen_images:
        continue

    bbox = bboxes[id]  # xxyy
    bw = bbox[1] - bbox[0]
    bh = bbox[3] - bbox[2]
    bbox[3] = bbox[3] + 2 / 3 * bh  # expand box downward 2/3
    bw = bbox[1] - bbox[0]
    bh = bbox[3] - bbox[2]
    if bw > bh:  # need to pad height
        pad = (bw - bh) // 2
        bbox[2] -= pad
        bbox[3] += pad
    elif bh > bw:  # need to pad width
        pad = (bh - bw) // 2
        bbox[0] -= pad
        bbox[1] += pad
    image_path = os.path.join(image_root, name)
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    right = int(max(bbox[1] - w, 0))
    bottom = int(max(bbox[3] - h, 0))
    left = int(max(-bbox[0], 0))
    top = int(max(-bbox[2], 0))

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, None,
                               value=(127, 127, 127))

    image = image[int(bbox[2]) + top: int(bbox[3]) + top, int(bbox[0]) + left: int(bbox[1]) + left]
    image = cv2.resize(image, (128, 128), cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(save_dir, str(id) + "_" + name.replace("/", "_")), image)
    count += 1
print(count)
