import os
from scipy.io import loadmat

train_file_1 = open("./data/split/train_labeled_1_4.txt", 'r').readlines()
train_file_2 = open("./data/split/train_unlabeled_1_4.txt", 'r').readlines()

test_file = open('./data/split/test.txt', 'r').readlines()

train_annotations_path = '/home/s/chaunm/DATA/AFLW/train_128_4/annotations'
train_images_path = '/home/s/chaunm/DATA/AFLW/train_128_4/images'
test_annotations_path = '/home/s/chaunm/DATA/AFLW/test_128_4/annotations'
test_images_path = '/home/s/chaunm/DATA/AFLW/test_128_4/images'


def process_file(f, annotation_path, unsupervised=False):
    image_names = []
    for line in f:
        line = line.rstrip()
        mat_path = os.path.join(annotation_path, line)
        a = loadmat(mat_path)
        a['image_name'] = a['image_name'][0].split("_")[-1]
        image_names.append(a['image_name'])
    return image_names


train_annot_supervise = process_file(train_file_1, train_annotations_path, unsupervised=True)
train_annot_unsupervise = process_file(train_file_2, train_annotations_path, unsupervised=True)
test_annot = process_file(test_file, test_annotations_path, unsupervised=True)

for name in train_annot_supervise:
    if name in train_annot_unsupervise or name in test_annot:
        print("test 1: ", name)

for name in train_annot_unsupervise:
    if name in train_annot_supervise or name in test_annot:
        print("test 1: ", name)
