import numbers
import os
import shutil
from pathlib import Path
from scipy.io import loadmat, savemat
import numpy as np
import torch
import cv2
from tqdm import tqdm
from skimage import io
from PIL import Image
from threading import Thread
from argparse import ArgumentParser
import json


def generate_label_map(pts, height, width, sigma, downsample, nopoints, mask, ctype):
    # if isinstance(pts, numbers.Number):
    # this image does not provide the annotation, pts is a int number representing the number of points
    # return np.zeros((height,width,pts+1), dtype='float32'), np.ones((1,1,1+pts), dtype='float32')
    # nopoints == True means this image does not provide the annotation, pts is a int number representing the number of points

    assert isinstance(pts, np.ndarray) and len(pts.shape) == 2 and pts.shape[0] == 3, 'The shape of points : {}'.format(
        pts.shape)
    if isinstance(sigma, numbers.Number):
        sigma = np.zeros((pts.shape[1])) + sigma
    assert isinstance(sigma, np.ndarray) and len(sigma.shape) == 1 and sigma.shape[0] == pts.shape[
        1], 'The shape of sigma : {}'.format(sigma.shape)

    offset = downsample / 2.0 - 0.5
    num_points, threshold = pts.shape[1], 0.01

    if nopoints == False:
        visiable = mask.astype('bool')
    else:
        visiable = (mask * 0).astype('bool')
    # assert visiable.shape[0] == num_points

    transformed_label = np.fromfunction(lambda y, x, pid: ((offset + x * downsample - pts[0, pid]) ** 2 \
                                                           + (offset + y * downsample - pts[1, pid]) ** 2) \
                                                          / -2.0 / sigma[pid] / sigma[pid],
                                        (height, width, num_points), dtype=int)
    distance = np.fromfunction(lambda y, x, pid: ((offset + x * downsample - pts[0, pid]) ** 2 \
                                                  + (offset + y * downsample - pts[1, pid]) ** 2) \
                               ,
                               (height, width, num_points), dtype=int)
    mask_distance = distance < (sigma * 3) ** 2
    mask_heatmap = np.ones((1, 1, num_points + 1), dtype='float32')
    mask_heatmap[0, 0, :num_points] = visiable
    mask_heatmap[0, 0, num_points] = (nopoints == False)

    if ctype == 'laplacian':
        transformed_label = (1 + transformed_label) * np.exp(transformed_label)
    elif ctype == 'gaussian':
        transformed_label = np.exp(transformed_label)
    else:
        raise TypeError('Does not know this type [{:}] for label generation'.format(ctype))
    transformed_label[transformed_label < threshold] = 0
    transformed_label[transformed_label > 1] = 1
    # transformed_label = transformed_label * mask_heatmap[:, :, :num_points]
    transformed_label = transformed_label * mask  # not masking H x W x C
    # print(norm.shape, norm[15])
    transformed_label = transformed_label
    background_label = 1 - np.amax(transformed_label, axis=2)
    background_label[background_label < 0] = 0
    heatmap = np.concatenate((transformed_label, np.expand_dims(background_label, axis=2)), axis=2).astype('float32')

    return heatmap, mask_heatmap


root = "/home/s/chaunm/DATA/AFLW"
save_size = (64, 64)
num_cpu = 8
pth_file = "/home/s/chaunm/DATA/AFLW/neck-new-sep/train.pth"
pth_test_file = "/home/s/chaunm/DATA/AFLW/neck-new-sep/test.pth"
path_annotation = torch.load(pth_file, map_location='cpu')
path_test_annotation = torch.load(pth_test_file, map_location='cpu')
all_annotation = path_annotation + path_test_annotation
sep_id = len(path_annotation)
partition = len(all_annotation) // num_cpu
train_save_dir = '/home/s/chaunm/DATA/AFLW/train_neck_64_3'
test_save_dir = '/home/s/chaunm/DATA/AFLW/test_neck_64_3'
if os.path.exists(train_save_dir):
    shutil.rmtree(train_save_dir)
    shutil.rmtree(test_save_dir)
for dir in ['images', 'annotations']:
    Path(os.path.join(train_save_dir, dir)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(test_save_dir, dir)).mkdir(parents=True, exist_ok=True)


def generate(annot_list, start, end):
    for i in tqdm(range(start, end), total=end - start):
        a = annot_list[i]
        name = '/'.join(a['current_frame'].split('/')[-2:])
        image = cv2.imread(os.path.join(root, name))
        h, w, _ = image.shape

        landmark = a['points']
        bbox = a['box-GTB']
        label = a['label']
        if label == 1:
            prefix = 'OK'
        else:
            prefix = 'NG'
        bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        box_w, box_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        box_pad_horizon, box_pad_vertical = max(box_w, box_h) - box_w, max(box_w, box_h) - box_h
        bbox[0] -= box_pad_horizon // 2
        bbox[2] += box_pad_horizon // 2
        bbox[1] -= box_pad_vertical // 2
        bbox[3] += box_pad_vertical // 2
        # print(bbox[2] - bbox[0], bbox[3] - bbox[1])
        right = int(max(bbox[2] - w, 0))
        bottom = int(max(bbox[3] - h, 0))
        left = int(max(-bbox[0], 0))
        top = int(max(-bbox[1], 0))

        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, None,
                                   value=(127, 127, 127))
        image = image[int(bbox[1]) + top: int(bbox[3]) + top, int(bbox[0]) + left: int(bbox[2]) + left]

        bh, bw, _ = image.shape
        image = cv2.resize(image, save_size, cv2.INTER_LINEAR)

        landmark = landmark.transpose(1, 0)  # 5 x 3
        lm = landmark[:, :2]
        mask = np.sum(lm, axis=1) > 20

        landmark[:, 0] = (landmark[:, 0] - bbox[0]) / bw * save_size[0]
        landmark[:, 1] = (landmark[:, 1] - bbox[1]) / bw * save_size[1]
        # mask = landmark[2, :]
        heatmap, mask_heatmap = generate_label_map(
            landmark.copy().astype(int).transpose(1, 0),
            save_size[1], save_size[0], 3, 1, False, mask=mask,
            ctype='gaussian'
        )
        mask_heatmap = mask_heatmap[0][0]
        new_name = prefix + "_" + str(i) + "_" + name.replace("/", "_")
        save_dict = {
            "landmark": landmark[:, :2],
            "heatmap": heatmap,
            "mask_landmark": landmark[:, 2],
            "mask_heatmap": mask_heatmap,
            "headpose": np.array([0., 0., 0.]),
            "image_name": new_name
        }
        if i >= sep_id:
            save_dir = test_save_dir
        else:
            save_dir = train_save_dir
        cv2.imwrite(os.path.join(save_dir, 'images', new_name), image)
        savemat(os.path.join(save_dir, 'annotations', f'{prefix}_{i}.mat'), save_dict)
        # x = landmark[:, 0].astype(int)
        # y = landmark[:, 1].astype(int)
        # for j in range(5):
        #     point = (x[j], y[j])
        #     image = cv2.circle(image, point, 1, (255, 255, 0), 2)
        # hm_save = np.max(heatmap[:, :, :-1], axis=2, keepdims=True)
        # hm_save = np.concatenate([hm_save, hm_save, hm_save], axis=2)
        # hm_save = (hm_save * 255.0).astype(np.uint8)
        # cv2.imwrite(os.path.join(save_dir, f'{i}.png'), np.concatenate([image, hm_save], axis=0))
        # # print(landmark[:, 2])


if __name__ == "__main__":
    threads = []
    for p in range(num_cpu + 1):
        start = p * partition
        end = min((p + 1) * partition, len(all_annotation))
        print(start, end, len(all_annotation))
        thread = Thread(target=generate, args=(all_annotation, start, end))
        threads.append(thread)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
