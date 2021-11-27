from collections import OrderedDict
from enum import Enum

import timm
from timm.models.features import FeatureHooks
from torch.nn import Module, Conv2d
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

from model_definition.anchor_utils import AnchorGenerator
from model_definition.faster_rcnn import FasterRCNN
from models.object_detector import ObjectDetector


class Network(Enum):
    EFFICIENT_NET_V2 = 'tf_efficientnetv2_s_in21ft1k'
    MOBILE_NET_V3 = 'mobilenetv3_large_100_miil'


class PretrainedEfficientNetV2(ObjectDetector):

    def define_model(self):
        out_indices = (3, 4)
        feature_extractor = timm.create_model(
            Network.MOBILE_NET_V3.value,
            pretrained=True,
            features_only=True,
            out_indices=out_indices
        )
        out_channels = 256

        hooks = [
            {'module': 'blocks.5.0'},
            {'module': 'blocks.6.0'}
        ]
        feature_extractor.feature_hooks = FeatureHooks(hooks, feature_extractor.named_modules())

        # Freeze similarly to pytorch model.
        for child in list(feature_extractor.children())[:-1]:
            for p in child.parameters():
                p.requires_grad_(False)

        for p in list(feature_extractor.children())[-1][:3].parameters():
            p.requires_grad_(False)

        backbone = TimmBackboneWithFPN(
            backbone=feature_extractor,
            in_channels_list=[160, 960],
            out_channels=out_channels
        )
        # backbone = TimmBackbone(feature_extractor, feature_extractor.feature_info.info[-1]['num_chs'], out_channels)

        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),) * (len(out_indices) + 1),
            aspect_ratios=((0.5, 1.0, 2.0),) * (len(out_indices) + 1)
        )

        roi_pooler = MultiScaleRoIAlign(
            featmap_names=list(range(len(out_indices))),
            # featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )
        return FasterRCNN(
            backbone,
            num_classes=self.num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            rpn_pre_nms_top_n_test=150,
            rpn_post_nms_top_n_test=150,
            rpn_score_thresh=0.05,
            min_size=320,
            max_size=640,
        )


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
