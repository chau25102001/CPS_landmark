import os
from scipy.io import loadmat
import numpy as np

train_file = open("./data/split/train_labeled_1_1.txt", 'r').readlines()
test_file = open('./data/split/test.txt', 'r').readlines()

train_annotations_path = '/home/s/chaunm/DATA/AFLW/train_128_4/annotations'
train_images_path = '/home/s/chaunm/DATA/AFLW/train_128_4/images'
test_annotations_path = '/home/s/chaunm/DATA/AFLW/test_128_4/annotations'
test_images_path = '/home/s/chaunm/DATA/AFLW/test_128_4/images'

train_file_neck = open("./data/split_neck_paper/train_labeled_1_1.txt", 'r').readlines()
test_file_neck = open('./data/split_neck_paper/test.txt', 'r').readlines()

train_annotations_path_neck = '/home/s/chaunm/DATA/AFLW/train_neck_paper_128_4/annotations'
train_images_path_neck = '/home/s/chaunm/DATA/AFLW/train_neck_paper_128_4/images'
test_annotations_path_neck = '/home/s/chaunm/DATA/AFLW/test_neck_paper_128_4/annotations'
test_images_path_neck = '/home/s/chaunm/DATA/AFLW/test__neck_paper_128_4/images'


def process_file(f, annotation_path, unsupervised=False):
    image_names = []
    landmarks = []
    for line in f:
        line = line.rstrip()
        mat_path = os.path.join(annotation_path, line)
        a = loadmat(mat_path)
        # print(a.keys())
        a['image_name'] = "/".join(a['image_name'][0].split("_")[-2:])
        image_names.append(a['image_name'])
        landmarks.append(a['landmark'])
    return image_names, landmarks


train_annot_supervise_1, landmark_train_1 = process_file(train_file, train_annotations_path, unsupervised=True)
train_annot_supervise_2, landmark_train_2 = process_file(train_file_neck, train_annotations_path_neck,
                                                         unsupervised=True)
test_annot_1, landmark_test_1 = process_file(test_file, test_annotations_path, unsupervised=True)
test_annot_2, landmark_test_2 = process_file(test_file_neck, test_annotations_path_neck, unsupervised=True)

selected_image_names = []
temp_file = "./data/mae_image_neck.txt"
with open(temp_file, 'w') as f:
    for train_name in train_annot_supervise_1:
        if train_name not in test_annot_2:
            f.write(train_name + "\n")
