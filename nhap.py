import torch
import torch.nn as nn


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
    unfreeze_bn(norm)
    out = norm(a)
    print(norm.running_mean)