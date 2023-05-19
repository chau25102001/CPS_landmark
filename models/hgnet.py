from mmpose.models import HourglassNet
import torch
import torch.nn as nn
import torch.nn.functional as F


class HGNet(nn.Module):
    def __init__(self, num_stacks=2, feat_channel=256, num_classes=20, **kwargs):
        super().__init__()

        self.model = HourglassNet(num_stacks=num_stacks, feat_channel=feat_channel, downsample_times=3)
        self.head = nn.Conv2d(feat_channel, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        _, _, h, w = x.shape
        feat = self.model(x)
        output = self.head(feat[-1])
        output = F.interpolate(output, (h, w), mode='bilinear')
        return output, None


if __name__ == "__main__":
    model = HGNet()
    count = 0
    for p in model.parameters():
        count += p.numel()
    print(count)

    a = torch.randn((1, 3, 64, 64))
    output, _ = model(a)
    print(output.shape)
