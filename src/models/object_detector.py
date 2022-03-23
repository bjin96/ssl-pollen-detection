import timm
from pytorch_lightning import LightningModule
from torchmetrics.detection.map import MeanAveragePrecision
from torchvision.ops import MultiScaleRoIAlign

from src.model_definition.anchor_utils import AnchorGenerator
from src.model_definition.faster_rcnn import FasterRCNN
from src.models.timm_adapter import Network, TimmBackboneWithFPN


class ObjectDetector(LightningModule):

    def __init__(
            self,
            num_classes: int,
            batch_size: int,
            timm_model: Network,
            min_image_size: int,
            max_image_size: int,
            freeze_backbone: bool = False,
    ):
        """
        Creates a Faster R-CNN model with a pre-trained backbone from timm
        (https://github.com/rwightman/pytorch-image-models) and feature pyramid network.

        Args:
            num_classes: Number of classes to classify objects in the image. Includes an additional background class.
            batch_size: Size of the batch per training step.
            timm_model: Identifier for a pre-trained timm backbone.
            min_image_size: Minimum size to which the image is scaled.
            max_image_size: Maximum size to which the image is scaled.
            freeze_backbone: Whether to freeze the backbone for the training.
        """
        super().__init__()
        self.num_classes = num_classes
        self.timm_model = timm_model
        self.freeze_backbone = freeze_backbone
        self.model = self.define_model(min_image_size, max_image_size)
        self.validation_mean_average_precision = MeanAveragePrecision(class_metrics=True)
        self.test_mean_average_precision = MeanAveragePrecision(class_metrics=True)
        self.batch_size = batch_size

    def define_model(self, min_image_size, max_image_size):
        feature_extractor = timm.create_model(
            self.timm_model.value,
            pretrained=True,
            features_only=True,
        )
        out_indices = feature_extractor.feature_info.out_indices
        out_channels = 256
        in_channels = [i['num_chs'] for i in feature_extractor.feature_info.info]

        if self.freeze_backbone:
            # Freeze similarly to pytorch model.
            for child in list(feature_extractor.children())[:-1]:
                for p in child.parameters():
                    p.requires_grad_(False)

            for p in list(feature_extractor.children())[-1][:3].parameters():
                p.requires_grad_(False)

        backbone = TimmBackboneWithFPN(
            backbone=feature_extractor,
            in_channels_list=in_channels,
            out_channels=out_channels
        )

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
            min_size=min_image_size,
            max_size=max_image_size,
        )

    def forward(self, images, targets=None, teacher_box_predictor=None, unsupervised_loss_weight=1.0):
        return self.model(images, targets, teacher_box_predictor, unsupervised_loss_weight)

