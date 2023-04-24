import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import math


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        # self.bn1 = nn.GroupNorm(4, middle_channels)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        # self.bn2 = nn.GroupNorm(4, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class MyFlatten(nn.Module):
    def __init__(self):
        super(MyFlatten, self).__init__()

    def forward(self, x):
        return x[..., 0, 0]


class UnetDecoder(nn.Module):
    def __init__(self, embed_dims, n_classes):
        super().__init__()
        embed_dims = sorted(embed_dims, reverse=True)
        embed_dims_pad = [0] + embed_dims
        self.blocks = nn.ModuleList([
            VGGBlock(embed_dims_pad[i] + embed_dims_pad[i + 1], embed_dims_pad[i + 1], embed_dims_pad[i + 1])
            for i in range(len(embed_dims_pad) - 1)
        ])
        self.last_layer = nn.Conv2d(embed_dims[-1], n_classes, kernel_size=1)
        # self.class_head = nn.Sequential(
        #     nn.Conv2d(embed_dims[0], embed_dims[0] // 4, 3, stride=1, padding=1, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(embed_dims[0] // 4),
        #     nn.Conv2d(embed_dims[0] // 4, embed_dims[0] // 4, 3, stride=1, padding=1, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm2d(embed_dims[0] // 4),
        #     nn.AdaptiveAvgPool2d(1),
        #     MyFlatten(),
        #     nn.Linear(embed_dims[0] // 4, n_classes - 1, bias=True)
        # )
        self.class_head = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                        MyFlatten(),
                                        nn.Linear(embed_dims[0], n_classes - 1, bias=True))
        # self.class_head = nn.Linear(embed_dims[0], n_classes - 1, bias=False)  # for keypoint classification
        self.dummy = nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, xs):
        x = xs[0]
        last_feature = self.dummy(x)
        for i in range(len(self.blocks)):
            if i == 0:
                x = self.blocks[i](xs[i])
            else:
                x = self.blocks[i](torch.cat(
                    [F.interpolate(x, size=xs[i].shape[-2:], mode='bilinear', align_corners=True), xs[i]],
                    dim=1))
        x = self.last_layer(x)
        visible = self.class_head(last_feature)
        # print(visible.shape)
        return x, visible
