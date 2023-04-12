import math, numbers
import os
import shutil

from scipy.io import loadmat
from pprint import pprint
import numpy as np
import torch
import cv2
import albumentations as A

transforms = A.Compose([
    A.HorizontalFlip(p=1),
    A.Rotate(limit=40, always_apply=True)
])
path = r'/home/s/chaunm/DATA/AFLW/AFLWinfo_release.mat'
image_path = '/home/s/chaunm/DATA/AFLW'
annotations = loadmat(path)
# pprint(annotations)
bbox = annotations['bbox']
landmark = annotations['data']
mask = annotations['mask_new']
names = np.array([n[0][0] for n in annotations['nameList']])
ra = annotations['ra']
print(names[ra[0][1025]])  # print(bbox.shape)
# print(landmark.shape)
# print(mask.shape)
# print(bbox[1])
# print(landmark[1])
# print(mask[1])
extra_annot = torch.load(r'/home/s/chaunm/DATA/AFLW/aflw-sqlite.pth')
# print(names[1][0][0])
save_dir = "./sample_images"
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.mkdir(save_dir)


# print(extra_annot['3/image00188.jpg'])

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
    transformed_label = transformed_label * mask_heatmap[:, :, :num_points]

    background_label = 1 - np.amax(transformed_label, axis=2)
    background_label[background_label < 0] = 0
    heatmap = np.concatenate((transformed_label, np.expand_dims(background_label, axis=2)), axis=2).astype('float32')

    return heatmap * mask_heatmap, mask_heatmap


# for i in range(20, 30):
#     image_name = names[i][0][0]
#     box_i = bbox[i]
#     landmark_i = landmark[i].reshape(2, 19).transpose(1, 0)
#     max_x = np.max(landmark_i[:, 0])
#     max_y = np.max(landmark_i[:, 1])
#     min_x = np.min(landmark_i[:, 0])
#     min_y = np.min(landmark_i[:, 1])
#     mask_i = mask[i]
#     extra_annot_i = np.array(extra_annot[image_name])
#     pose = [0, 0, 0]
#     # print(box_i)
#     for f in extra_annot_i:
#         if f['rect'][0] == box_i[0] and f['rect'][1] == box_i[2]:
#             pose = f['pose'][0]
#             # print(f['rect'])
#     pose = np.array(pose)
#     image = cv2.imread(os.path.join(image_path, image_name))
#     h, w, _ = image.shape
#     pad_right = int(max(box_i[1] - w, max_x - w, 0))
#     pad_bottom = int(max(box_i[3] - h, max_y - h, 0))
#     pad_left = int(max(-box_i[0], -min_x, 0))
#     pad_top = int(max(-box_i[2], -min_y, 0))
#     # print(pad_right, pad_bottom, box_i[1] - w)
#     image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, None,
#                                value=(127, 127, 127))
#     # print(image.shape)
#     # print(landmark_i)
#     # crop image
#     image = image[int(box_i[2]) + pad_top:int(box_i[3]) + pad_top, int(box_i[0]) + pad_left: int(box_i[1]) + pad_left]
#     bh, bw, _ = image.shape
#
#     image = cv2.resize(image, (256, 256))
#     # image = cv2.rectangle(image, (int(box_i[0]) + pad_left, int(box_i[2]) + pad_top),
#     #                       (int(box_i[1]) + pad_left, int(box_i[3]) + pad_top),
#     #                       (255, 0, 0), 2)
#     landmark_i[:, 0] = (landmark_i[:, 0] - box_i[0]) / bw * 256
#     landmark_i[:, 1] = (landmark_i[:, 1] - box_i[2]) / bh * 256
#     lm_test = landmark_i.copy().transpose(1, 0)
#     # lm_test2 = lm_test.copy()
#     # lm_test[0, :] /= bw
#     # lm_test[1, :] /= bh
#     # print(lm_test)
#     heatmap, _ = generate_label_map(np.concatenate([lm_test, np.expand_dims(mask_i, 0)], axis=0), 256,
#                                     256, 3, 1, False,
#                                     mask_i, 'gaussian')
#     out = transforms(image=image, mask=heatmap)
#     image, heatmap = out['image'], out['mask']
#     # print(heatmap[:, :, 1])
#     # point = 19
#     # hm_save = np.expand_dims(heatmap[:, :, 0], 2)
#     # heatmap = np.flip(heatmap, axis=1)
#     hm_save = np.sum(heatmap[:, :, :-1], axis=2, keepdims=True)
#     hm_save = np.concatenate([hm_save, hm_save, hm_save], axis=2)
#     hm_save = (hm_save * 255.0).astype(np.uint8)
#     # cv2.imwrite(os.path.join(save_dir, str(i) + "_hm_" + image_name.replace("/", "_")), hm_save)
#     for p in range(0, 19):
#         point = (int(landmark_i[p][0]), int(landmark_i[p][1]))
#         image = cv2.circle(image, point, 1, (255, 0, 0), 2)
#         # print(point)
#         # print(mask_i[p])
#     # print(mask_i)
#     # print(image_name, pose * 180 / math.pi)
#     cv2.imwrite(os.path.join(save_dir, str(i) + "_" + image_name.replace("/", "_")),
#                 np.concatenate([image, hm_save], axis=0))
print(extra_annot['3/image21423.jpg'])
