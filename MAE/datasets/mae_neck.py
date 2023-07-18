import os
import cv2
import numpy as np
import torch.utils.data
import torch.utils.data as data
import albumentations as A
from scipy.io import loadmat
from termcolor import colored
from utils.utils import InfiniteDataLoader
from albumentations.augmentations.geometric.rotate import Rotate
from albumentations.augmentations.geometric import functional as F
from albumentations.augmentations.crops import functional as FCrops


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
            # RandomRotate(30, p=0.5),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.GaussianBlur(p=0.2, blur_limit=3),
            A.ToGray(p=0.5), ])

    def __call__(self, image):
        output = self.transforms(image=image)
        image = output['image']
        image = normalize(image, self.mean, self.std)
        image = image.transpose(2, 0, 1).astype(np.float32)
        return image


class ValPreprocessing:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = normalize(image, self.mean, self.std)
        image = image.transpose(2, 0, 1).astype(np.float32)
        return image


class MAE_AFLW_NECK(data.Dataset):
    def __init__(self, text_annot_file, image_path, training=True, mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        super().__init__()
        self.text_annot_file = text_annot_file
        self.image_path = image_path
        self.training = training
        self.transforms = TrainPreprocessing(mean, std) if training else ValPreprocessing(mean, std)
        self.mean = mean
        self.std = std
        self.names = [n.strip() for n in open(text_annot_file, 'r').readlines()]
        print(colored(f"{len(self.names)} images", 'red'))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        image_name = self.names[index]
        image = cv2.imread(os.path.join(self.image_path, image_name))
        image = self.transforms(image)
        return {"image": image}


def get_train_loader(config, **kwargs):
    dataset = MAE_AFLW_NECK(config.train_text,
                            config.images_path,
                            training=True,
                            mean=config.mean,
                            std=config.std)
    dataloader = InfiniteDataLoader(dataset,
                                    batch_size=config.train_batch_size,
                                    num_workers=config.num_workers,
                                    shuffle=True,
                                    drop_last=True,
                                    pin_memory=True)
    return dataloader


def get_test_loader(config, **kwargs):
    dataset = MAE_AFLW_NECK(config.test_text,
                            config.images_path,
                            training=False,
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
