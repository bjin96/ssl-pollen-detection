from collections import OrderedDict
from enum import Enum

from torch.nn import Module, Conv2d
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool


class Network(Enum):
    EFFICIENT_NET_V2 = 'tf_efficientnetv2_s_in21ft1k'
    MOBILE_NET_V3 = 'mobilenetv3_large_100_miil'
    RESNET_50 = 'resnet50'


class TimmBackboneWithFPN(Module):
    def __init__(self, backbone, in_channels_list, out_channels, extra_blocks=None):
        super().__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.body = backbone
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = OrderedDict([(index, value) for index, value in enumerate(x)])
        x = self.fpn(x)
        return x


class TimmBackbone(Module):

    def __init__(self, backbone, in_channels, out_channels):
        super().__init__()
        self.backbone = backbone
        self.conv = Conv2d(in_channels, out_channels, 1)
        self.out_channels = out_channels

    def forward(self, x):
        y = self.backbone(x)[0]
        y = self.conv(y)
        return y
