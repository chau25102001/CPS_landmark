import math

import torch.utils.data as data
import torch
from config.config import get_config
import torch.nn as nn
import numpy as np


# config = get_config(train=False)

def generate_pseudo(heatmap, mask_size):
    c, h, w = heatmap.size(0), heatmap.size(1), heatmap.size(2)
    # print(c,h,w)

    score, index = torch.max(heatmap.view(c, -1), 1)
    index_w = index % w
    index_h = index // h

    mask = torch.zeros_like(heatmap)
    for i in range(c):
        mask[i, max(index_h[i] - mask_size, 0):min(index_h[i] + mask_size + 1, h),
        max(index_w[i] - mask_size, 0):min(index_w[i] + mask_size + 1, w)] = 1
    pesudo = heatmap * mask

    pesudo = torch.concat([pesudo[:-1, :, :], heatmap[-1, :, :].unsqueeze(0)], dim=0)

    return pesudo


def generate_gaussian(radius=12, sigma=4):
    size = 2 * radius + 1
    center = radius
    heatmap = np.fromfunction(lambda y, x: ((x - center) ** 2 + (y - center) ** 2) \
                                           / -2.0 / sigma ** 2,
                              (size, size), dtype=int)

    transformed_label = np.exp(heatmap)
    transformed_label = torch.from_numpy(transformed_label)
    return transformed_label


def generate_pseudo_label(heatmap, mask):
    '''
    :param heatmap: c x h x w
    :param mask: h x w
    :return:
    '''
    c, h, w = heatmap.size(0), heatmap.size(1), heatmap.size(2)
    score, index = torch.max(heatmap.view(c, -1), 1)
    index_w = (index % w).int()
    index_h = (index // h).int()
    zero_mask = torch.zeros_like(heatmap, device=heatmap.device, dtype=heatmap.dtype)
    mask_size = (mask.size(0) - 1) // 2  # radius

    for i in range(c - 1):  # dont generate background
        h_peak = index_h[i]
        w_peak = index_w[i]
        start_y = max(h_peak - mask_size, 0)
        end_y = min(h_peak + mask_size + 1, h)
        start_x = max(w_peak - mask_size, 0)
        end_x = min(w_peak + mask_size + 1, w)
        print(start_y, end_y, start_x, end_x, i)
        zero_mask[i, start_y:end_y, start_x:end_x] = mask[
                                                     mask_size - h_peak + start_y:mask_size - h_peak + end_y,
                                                     mask_size - w_peak + start_x:mask_size - w_peak + end_x]
    positive_mask = torch.amax(zero_mask[:-1, :, :], dim=0)
    zero_mask[-1, :, :] = 1 - positive_mask
    return zero_mask


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

    def accumulate(self):
        return self.sum


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


class InfiniteDataLoader(data.DataLoader):
    def __init__(self, *args, **kwargs):
        super(InfiniteDataLoader, self).__init__(*args, **kwargs)
        self.iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = super().__iter__()
            batch = next(self.iterator)
        return batch


# pool = torch.nn.MaxPool1d(kernel_size=config.img_height * config.img_width, stride=config.img_height * config.img_width,
#                           return_indices=True)


def heatmap2coordinate(heatmap):
    '''heatmap: Bsize x (K+1) x H x W'''
    b, k, h, w = heatmap.shape
    heatmap = heatmap.view(b, k, -1)
    index = torch.argmax(heatmap, dim=-1, keepdim=True)
    # _, index = torch.max(heatmap, dim=-1, keepdim=True)
    index = index[:, :-1, :]  # ignore background channel
    # print(index)
    xs = index % w
    ys = index // h
    landmarks = torch.cat([xs, ys], dim=-1)  # Bsize x K x 2
    return landmarks


def local_filter(logits, confidence_threshold=0.7):
    '''

    :param logits: Tensor(bsize x (K + 1) x H x W)
    :param confidence_threshold: float
    :return: mask: Tensor(bsize x K)
    '''
    b, k, h, w = logits.shape
    logits = logits.view(b, k, -1)
    max_values, _ = torch.max(logits, dim=-1, keepdim=True)  # bsize x (K + 1) x 1
    mask = torch.where(max_values >= confidence_threshold, 1., 0.)
    return mask


class Accuracy(torch.nn.Module):
    def __init__(self, threshold=0.5):
        super(Accuracy, self).__init__()
        self.threshold = threshold

    def forward(self, pred, gt):
        assert pred.shape == gt.shape, f'2 shapes must be equal, found :{pred.shape} and {gt.shape}'
        pred = torch.sigmoid(pred)
        pred = torch.where(pred > self.threshold, 1., 0.)
        matched = torch.eq(pred, gt)
        acc = torch.sum(matched) / pred.numel()
        return acc


class NME(torch.nn.Module):
    def __init__(self, h, w):
        super(NME, self).__init__()
        self.h = h
        self.w = w

    def forward(self, pred, gt, mask=None):
        '''

        :param pred: bsize x k x 2
        :param gt: bsize x k x 2
        :param mask: bsize x k
        :return: NME loss
        '''
        if len(pred.shape) == 4:  # heatmap
            pred = heatmap2coordinate(pred).float()
        if len(gt.shape) == 4:  # heatmap
            gt = heatmap2coordinate(gt).float()
        norm = gt.shape[1]
        if mask is None:
            mask = 1
        else:
            norm = torch.sum(mask, dim=1)

        loss = torch.sqrt(torch.sum((pred - gt) ** 2, dim=-1)) * mask / math.sqrt(self.w * self.h)
        loss = torch.sum(loss, dim=1) / norm * 100  # mean over num keypoints
        return torch.mean(loss)


def freeze_bn(model: torch.nn.Module):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False
            m.eval()


def unfreeze_bn(model: torch.nn.Module):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = True
            m.train()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    a = generate_gaussian().cpu().numpy()
    print(a.shape)
    a = (a * 255.0).astype(int)
    plt.imshow(a)
    plt.show()
