from models.decode_heads import *
from models.segformer import *
from models.resnet import *
from models.hgnet import *


class SingleNetwork(nn.Module):
    def __init__(self, num_classes, model_size='b3', model='segformer'):
        super(SingleNetwork, self).__init__()
        if model == 'segformer':
            self.backbone = eval(f'mit_{model_size}')()
        elif model == 'resnet':
            self.backbone = eval(f'ResNet{model_size}')()
        self.decoder = UnetDecoder(embed_dims=self.backbone.embed_dims[::-1], n_classes=num_classes)
        self.apply(self._init_weights)

    def forward(self, x):
        b, c, h, w = x.shape
        blocks = self.backbone(x)
        heatmap, visible = self.decoder(blocks[::-1])
        heatmap = F.interpolate(heatmap, size=(h, w), mode='bilinear', align_corners=True)
        return heatmap, visible

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


class Network(nn.Module):
    def __init__(self, num_classes, model_size, model):
        super(Network, self).__init__()
        if model == 'hgnet':
            module = HGNet
        else:
            module = SingleNetwork
        self.branch1 = module(num_classes, model_size, model)
        self.branch2 = module(num_classes, model_size, model)

    def forward(self, data, step=1):
        if step == 1:
            return self.branch1(data)
        elif step == 2:
            return self.branch2(data)


if __name__ == "__main__":
    model = SingleNetwork(21, model_size='18', model='resnet')
