import os
import cv2
import numpy as np
import torch.utils.data
import torch.utils.data as data
import albumentations as A
from scipy.io import loadmat
from termcolor import colored
from config.config import get_config
from utils.utils import InfiniteDataLoader

config = get_config(train=False)


def normalize(img, mean, std):
    img = img.astype(np.float32) / 255.0
    img = img - mean
    img = img / std

    return img


class TrainPreprocessing:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.transforms = A.Compose([
            A.Rotate(30, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.5),
            # A.HorizontalFlip(p=0.5),
            # A.RandomRotate90(p=0.5),
            A.ToGray(p=0.5),
        ])

    def __call__(self, image, heatmap=None):
        if heatmap is None:
            image = self.transforms(image=image)['image']
        else:
            output = self.transforms(image=image, mask=heatmap)
            image = output['image']
            heatmap = output['mask']
            heatmap = heatmap.transpose(2, 0, 1).astype(np.float32)

        image = normalize(image, self.mean, self.std)
        image = image.transpose(2, 0, 1).astype(np.float32)

        return image, heatmap


class ValPreprocessing:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, heatmap=None):
        image = normalize(image, self.mean, self.std)
        image = image.transpose(2, 0, 1).astype(np.float32)
        if heatmap is not None:
            heatmap = heatmap.transpose(2, 0, 1).astype(np.float32)

        return image, heatmap


def get_train_loader(unsupervised=False):
    if unsupervised:
        dataset = AFLW(config.train_text_unlabeled,
                       config.train_annotations_path,
                       config.train_images_path,
                       training=True,
                       unsupervised=unsupervised,
                       mean=config.mean,
                       std=config.std)
    else:
        dataset = AFLW(config.train_text_labeled,
                       config.train_annotations_path,
                       config.train_images_path,
                       training=True,
                       unsupervised=unsupervised,
                       mean=config.mean,
                       std=config.std)
    dataloader = InfiniteDataLoader(dataset,
                                    batch_size=config.train_batch_size,
                                    num_workers=config.num_workers,
                                    drop_last=True,
                                    shuffle=True,
                                    pin_memory=True
                                    )
    return dataloader


def get_test_loader():
    dataset = AFLW(config.test_text,
                   config.test_annotations_path,
                   config.test_images_path,
                   training=False,
                   unsupervised=False,
                   mean=config.mean,
                   std=config.std)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=config.test_batch_size,
                                             num_workers=config.num_workers,
                                             drop_last=False,
                                             shuffle=False,
                                             pin_memory=True
                                             )
    return dataloader


class AFLW(data.Dataset):
    def __init__(self, text_annot_file, annotation_path, image_path, training=True, unsupervised=False,
                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super(AFLW, self).__init__()
        self.text_annot_file = text_annot_file
        self.annotation_path = annotation_path
        self.image_path = image_path
        self.training = training
        self.unsupervised = unsupervised
        self.transforms = TrainPreprocessing(mean, std) if training else ValPreprocessing(mean, std)
        self.annotations = self.load_annotations_from_mat(text_annot_file)
        self.mean = mean
        self.std = std
        print(colored(f"{len(self.annotations)} images", 'red'))

    def load_annotations_from_mat(self, text_file):
        lines = open(text_file, 'r').readlines()
        annotations = []
        for line in lines:
            line = line.rstrip()
            mat_path = os.path.join(self.annotation_path, line)
            a = loadmat(mat_path)
            a['image_name'] = a['image_name'][0]
            if not self.unsupervised:
                annotations.append(a)
            else:
                annotations.append({'image_name': a['image_name']})
        annotations = np.array(annotations)
        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        annotation = self.annotations[index]
        image_name = annotation['image_name']
        heatmap = None
        image = cv2.imread(os.path.join(self.image_path, image_name))

        if not self.unsupervised:
            landmark = annotation['landmark']
            heatmap = annotation['heatmap']
            headpose = annotation['headpose']
            mask_heatmap = annotation['mask_heatmap']
            if self.transforms is not None:
                image, heatmap = self.transforms(image, heatmap)
            return {'image': image, 'heatmap': heatmap, 'landmark': landmark, 'headpose': headpose,
                    'mask_heatmap': mask_heatmap}
        else:
            if self.transforms is not None:
                image, heatmap = self.transforms(image, heatmap)
            return {'image': image}
