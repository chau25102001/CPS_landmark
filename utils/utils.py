import math

import torch.utils.data as data
import torch
from config.config import get_config
import torch.nn as nn

config = get_config(train=False)


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


pool = torch.nn.MaxPool1d(kernel_size=config.img_height * config.img_width, stride=config.img_height * config.img_width,
                          return_indices=True)


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
        loss = torch.mean(torch.sum(loss, dim=1) / norm) * 100  # mean over batch
        return loss


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
    norm = torch.nn.BatchNorm2d(num_features=3)
    a = torch.rand(5, 3, 128, 128)
    out = norm(a)
    print(norm.running_mean)
    out = norm(a)
    print(norm.running_mean)
    freeze_bn(norm)
    out = norm(a)
    print(norm.running_mean)
