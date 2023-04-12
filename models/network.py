from models.decode_heads import *
from models.segformer import *
from models.resnet import *


class SingleNetwork(nn.Module):
    def __init__(self, num_classes, model_size='b3', model='segformer'):
        super(SingleNetwork, self).__init__()
        if model == 'segformer':
            self.backbone = eval(f'mit_{model_size}')()
        elif model == 'resnet':
            self.backbone = eval(f'ResNet{model_size}')()
        self.decoder = UnetDecoder(embed_dims=self.backbone.embed_dims[::-1], n_classes=num_classes)

    def forward(self, x):
        b, c, h, w = x.shape
        blocks = self.backbone(x)
        pred = self.decoder(blocks[::-1])
        pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)
        return pred


class Network(nn.Module):
    def __init__(self, num_classes, model_size, model):
        super(Network, self).__init__()
        self.branch1 = SingleNetwork(num_classes, model_size, model)
        self.branch2 = SingleNetwork(num_classes, model_size, model)

    def forward(self, data, step=1):
        if step == 1:
            return self.branch1(data)
        elif step == 2:
            return self.branch2(data)


if __name__ == "__main__":
    model = SingleNetwork(21, model_size='18', model='resnet')
    a = torch.randn((2, 3, 512, 512))
    pred = model(a)
    print(pred.shape)
