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

parser = ArgumentParser(description='prepare data')
parser.add_argument("--root", type=str, help="path to AFLW folder")
args = parser.parse_args()
mat_path = os.path.join(args.root, 'AFLWinfo_release.mat')
image_path = args.root
train_root = Path(os.path.join(image_path, 'train_preprocessed'))
test_root = Path(os.path.join(image_path, 'test_preprocessed'))
if os.path.exists(train_root):
    shutil.rmtree(train_root)
    shutil.rmtree(test_root)
train_root.mkdir(parents=True, exist_ok=True)
test_root.mkdir(parents=True, exist_ok=True)
for dir in ['images', 'annotations']:
    Path(os.path.join(train_root, dir)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(test_root, dir)).mkdir(parents=True, exist_ok=True)

annotations = loadmat(mat_path)
extra_annotations = torch.load(os.path.join(args.root, 'aflw-sqlite.pth'))  # heapose and bbox

bboxes = annotations['bbox']
landmarks = annotations['data']
masks = annotations['mask_new']
names = np.array([n[0][0] for n in annotations['nameList']])
indices = annotations['ra'][0] - 1
train_indices = indices[:20000]
test_indices = indices[20000:]

num_annotations = landmarks.shape[0]
landmarks = landmarks.reshape(num_annotations, 2, 19).transpose(0, 2, 1)
save_size = (128, 128)


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
    transformed_label = transformed_label  # not masking H x W x C
    # print(norm.shape, norm[15])
    transformed_label = transformed_label
    background_label = 1 - np.amax(transformed_label, axis=2)
    background_label[background_label < 0] = 0
    heatmap = np.concatenate((transformed_label, np.expand_dims(background_label, axis=2)), axis=2).astype('float32')

    return heatmap, mask_heatmap
    # return heatmap, mask_heatmap  # not masking


debug = False
save_dir = "./sample_images"
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.mkdir(save_dir)
count = 0
total = len(indices)
num_cpu = 1
partition = total // num_cpu


def generate(ids, start):
    count = 0
    for c, index in tqdm(enumerate(ids), total=len(ids)):  # loop through all image
        image_name = names[index]
        bbox = bboxes[index]  # xmin, xmax, ymin, ymax
        landmark = landmarks[index]
        has_landmark = True
        if landmark is None:
            has_landmark = False
        mask = masks[index]
        try:
            extra = extra_annotations[image_name]

            pose = None
            for f in extra:
                if f['rect'][0] == bbox[0] and f['rect'][1] == bbox[2]:
                    pose = f['pose'][0]
            if pose is None:
                print(f"{image_name} has no headpose information")
                pose = np.array([0., 0., 0.])
            else:
                pose = np.array(pose)
        except KeyError:
            pose = np.array([0., 0., 0.])
        try:
            _ = io.imread(os.path.join(image_path, image_name))
            if image_name.endswith('.jpg'):
                i_test = Image.open(os.path.join(image_path, image_name))
                exif = i_test._getexif()
            image = cv2.imread(os.path.join(image_path, image_name))
        except Exception as e:
            print('skip this image', image_name, e)
            continue
        h, w, _ = image.shape
        bw, bh = bbox[1] - bbox[0], bbox[3] - bbox[2]
        pad_box_w = bw * 1 / 4
        pad_box_h = bh * 1 / 4  # pad each size 1/4
        bbox[0] -= pad_box_w
        bbox[1] += pad_box_w
        bbox[2] -= pad_box_h
        bbox[3] += pad_box_h
        right = int(max(bbox[1] - w, 0))
        bottom = int(max(bbox[3] - h, 0))
        left = int(max(-bbox[0], 0))
        top = int(max(-bbox[2], 0))

        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, None,
                                   value=(127, 127, 127))
        image = image[int(bbox[2]) + top: int(bbox[3]) + top, int(bbox[0]) + left: int(bbox[1]) + left]
        bh, bw, _ = image.shape

        image = cv2.resize(image, save_size, cv2.INTER_LINEAR)
        landmark[:, 0] = (landmark[:, 0] - bbox[0]) / bw * save_size[0]
        landmark[:, 1] = (landmark[:, 1] - bbox[2]) / bw * save_size[1]
        # landmark = landmark
        heatmap, mask_heatmap = generate_label_map(
            np.concatenate([landmark.copy().astype(int).transpose(1, 0), np.expand_dims(mask, 0)]),
            save_size[1], save_size[0], 4, 1, not has_landmark, mask, 'gaussian')
        mask_heatmap = mask_heatmap[0][0]
        new_name = str(index) + "_" + image_name.replace("/", "_")
        save_dict = {"landmark": landmark,
                     "heatmap": heatmap,
                     "mask_landmark": mask,
                     "mask_heatmap": mask_heatmap,
                     "headpose": pose,
                     "image_name": new_name}
        split = train_root if c + start < 20000 else test_root
        cv2.imwrite(os.path.join(split, 'images', new_name), image)
        savemat(os.path.join(split, 'annotations', f'{index}.mat'), save_dict)
        if debug:
            print(mask_heatmap)
            hm_save = np.max(heatmap[:, :, :-1], axis=2, keepdims=True)
            hm_save = np.concatenate([hm_save, hm_save, hm_save], axis=2)
            hm_save = (hm_save * 255.0).astype(np.uint8)
            for p in range(19):
                point = (int(landmark[p][0]), int(landmark[p][1]))
                image = cv2.circle(image, point, 1, (255, 0, 0), 2)
            cv2.imwrite(os.path.join(save_dir, str(index) + "_" + image_name.replace("/", "_")),
                        np.concatenate([image, hm_save], axis=0))
        if count > 10 and debug:
            break
        count += 1


threads = []
for p in range(num_cpu):
    part = indices[p * partition: min((p + 1) * partition, total)].copy()
    thread = Thread(target=generate, args=(part, p * partition))
    threads.append(thread)

for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
