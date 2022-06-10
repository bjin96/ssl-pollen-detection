import os
from copy import deepcopy
from typing import List

import pytorch_lightning as pl
import torch
import torchvision.transforms
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS, STEP_OUTPUT
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchmetrics.detection.map import MeanAveragePrecision

from src.data_loading.load_augsburg15 import Augsburg15DetectionDataset, collate_augsburg15_detection
from src.image_tools.overlap import clean_pseudo_labels
from src.models.exponential_moving_average import ExponentialMovingAverage
from src.models.object_detector import ObjectDetector, Augmentation, ClassificationLoss
from src.models.timm_adapter import Network
from src.training.transforms import Compose, ToTensor, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation


class SoftTeacher(pl.LightningModule):
    """
    Adapted SoftTeacher model from https://arxiv.org/abs/2106.09018.
    """

    def __init__(
            self,
            num_classes: int,
            batch_size: int,
            learning_rate: float,
            teacher_pseudo_roi_threshold: float,
            teacher_pseudo_rpn_threshold: float,
            student_inference_threshold: float,
            unsupervised_loss_weight: float,
            backbone: Network,
            min_image_size: int,
            max_image_size: int,
            augmentations: List[Augmentation],
            freeze_backbone: bool = False,
            classification_loss_function: ClassificationLoss = ClassificationLoss.CROSS_ENTROPY,
            student_only_epochs: int = 1,
    ):
        super(SoftTeacher, self).__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.batch_size = batch_size
        self.unsupervised_loss_weight = unsupervised_loss_weight
        self.learning_rate = learning_rate
        self.augmentations = augmentations
        self.student_only_epochs = student_only_epochs

        self.student = ObjectDetector(
            num_classes=num_classes,
            batch_size=batch_size,
            timm_model=backbone,
            min_image_size=min_image_size,
            max_image_size=max_image_size,
            freeze_backbone=freeze_backbone,
            classification_loss_function=classification_loss_function,
        )
        # Only use high confidence box predictions for inference.
        self.student.model.roi_heads.score_thresh = student_inference_threshold

        self.teacher = deepcopy(self.student)
        self.teacher.freeze()
        # Only use high confidence additional pseudo boxes.
        self.teacher.model.roi_heads.score_thresh = teacher_pseudo_roi_threshold
        self.teacher.model.rpn.score_thresh = teacher_pseudo_rpn_threshold
        self.teacher.eval()
        # TODO decay should change because student learning slows down https://arxiv.org/pdf/1703.01780.pdf.
        self.exponential_moving_average = ExponentialMovingAverage(
            self.student,
            self.teacher,
            ramp_up_decay=0.99,
            after_ramp_up_decay=0.999,
            ramp_up_epochs=3,
        )

        self.validation_mean_average_precision = MeanAveragePrecision(class_metrics=True, compute_on_step=False)
        self.test_mean_average_precision = MeanAveragePrecision(class_metrics=True, compute_on_step=False)

        self.student_augmenter = torchvision.transforms.Compose([
            # torchvision.transforms.RandomSolarize(threshold=float(torch.rand(1).numpy()), p=0.25),
            # torchvision.transforms.RandomApply(
            #     [torchvision.transforms.ColorJitter(brightness=(0., 1.), contrast=(0., 1.))],
            #     p=0.25
            # ),
            torchvision.transforms.RandomAdjustSharpness(sharpness_factor=float(torch.rand(1)), p=0.25),
        ])

    def on_before_zero_grad(self, optimizer: Optimizer) -> None:
        self.exponential_moving_average.update_teacher(self.current_epoch)

    def forward(self, x, y=None, teacher_box_predictor=None):

        y_labelled = self.student(x, y, teacher_box_predictor)

        # Box jittering: use box regression head of teacher (multiple times with different starting points) and look if
        # it comes to the same result (-> reliable regression, higher weight). Won't use here, because location
        # accuracy is not as important
        return y_labelled

    def train(self: 'SoftTeacher', mode: bool = True) -> 'SoftTeacher':
        super().train(mode)
        self.teacher.eval()
        return self

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, targets = batch

        # Consistency regularization: Use a weakly augmented batch for the teacher and a strongly augmented batch for
        # the student https://arxiv.org/pdf/2001.07685.pdf.

        student_images = self.student_augmenter(images)

        if self.current_epoch < self.student_only_epochs:
            raw_x_pseudo = [
                {
                    'boxes': torch.tensor([], dtype=targets[0]['boxes'].dtype, device=targets[0]['boxes'].device),
                    'labels': torch.tensor([], dtype=targets[0]['labels'].dtype, device=targets[0]['labels'].device),
                    'scores': torch.tensor([], dtype=targets[0]['boxes'].dtype, device=targets[0]['boxes'].device),
                } for _ in range(self.batch_size)
            ]
        else:
            # Originally, this would be two different batches, labelled + unlabelled.
            raw_x_pseudo = self.teacher(images, is_teacher=True)

        cleaned_y_pseudo = clean_pseudo_labels(raw_x_pseudo, targets)
        loss_dict = self(student_images, cleaned_y_pseudo, self.teacher.model.roi_heads.box_predictor)

        loss_dict['unsupervised_loss_classifier'] *= self.unsupervised_loss_weight
        total_loss = sum(loss for loss in loss_dict.values())
        self.log_dict(loss_dict, on_step=True, batch_size=self.batch_size)
        return total_loss

    def validation_step(self, batch, batch_idx) -> None:
        images, targets = batch
        predictions = self(images, targets)
        self.validation_mean_average_precision(predictions, targets)

    def on_validation_epoch_end(self) -> None:
        metrics = self.validation_mean_average_precision.compute()
        self._log_metrics(metrics, mode='validation')
        self.validation_mean_average_precision.reset()

    def test_step(self, batch, batch_idx):
        images, targets = batch
        predictions = self(images, targets)
        self.test_mean_average_precision(predictions, targets)

    def on_test_epoch_end(self) -> None:
        metrics = self.test_mean_average_precision.compute()
        self._log_metrics(metrics, mode='test')
        self.test_mean_average_precision.reset()

    def _log_metrics(self, mean_average_precision, mode):
        for index, value in enumerate(mean_average_precision['map_per_class']):
            mean_average_precision[f'map_per_class_{index}'] = value
        for index, value in enumerate(mean_average_precision['mar_100_per_class']):
            mean_average_precision[f'mar_100_per_class_{index}'] = value
        del mean_average_precision['map_per_class']
        del mean_average_precision['mar_100_per_class']
        for name, metric in mean_average_precision.items():
            self.log(f'{mode}_{name}', metric, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': ReduceLROnPlateau(optimizer, factor=0.5, patience=4),
                'monitor': 'validation_map_50',
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: Optimizer, optimizer_idx: int):
        optimizer.zero_grad(set_to_none=True)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        transforms_list = [ToTensor()]

        if Augmentation.HORIZONTAL_FLIP in self.augmentations:
            transforms_list.append(RandomHorizontalFlip(0.5))
        if Augmentation.VERTICAL_FLIP in self.augmentations:
            transforms_list.append(RandomVerticalFlip(0.5))

        if Augmentation.ROTATION in self.augmentations and Augmentation.ROTATION_CUTOFF in self.augmentations:
            raise ValueError("Cannot apply rotation and rotation cutoff data augmentation at the same time.")

        if Augmentation.ROTATION in self.augmentations:
            transforms_list.append(RandomRotation(0.5, 25, (1280, 960))),
        elif Augmentation.ROTATION_CUTOFF in self.augmentations:
            transforms_list.append(RandomRotation(0.5, 25, (1280, 960), True))

        train_dataset = Augsburg15DetectionDataset(
            root_directory=os.path.join(os.path.dirname(__file__), '../../datasets/pollen_only'),
            image_info_csv='pollen15_train_annotations_preprocessed.csv',
            transforms=Compose(transforms_list)
        )
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_augsburg15_detection,
            drop_last=True,
            shuffle=True,
            num_workers=2
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        validation_dataset = Augsburg15DetectionDataset(
            root_directory=os.path.join(os.path.dirname(__file__), '../../datasets/pollen_only'),
            image_info_csv='pollen15_val_annotations_preprocessed.csv',
            transforms=ToTensor()
        )
        return DataLoader(
            validation_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_augsburg15_detection,
            drop_last=True,
            num_workers=2
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        validation_dataset = Augsburg15DetectionDataset(
            root_directory=os.path.join(os.path.dirname(__file__), '../../datasets/pollen_only'),
            image_info_csv='pollen15_test_annotations_preprocessed.csv',
            transforms=ToTensor()
        )
        return DataLoader(
            validation_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_augsburg15_detection,
            drop_last=True,
            num_workers=2
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass
