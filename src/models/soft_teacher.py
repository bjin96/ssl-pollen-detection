import os
from copy import deepcopy
from typing import Optional

import pytorch_lightning as pl
import torch
import torchvision.transforms
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS, STEP_OUTPUT
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader
from torchmetrics.detection.map import MeanAveragePrecision

from src.data_loading.load_augsburg15 import Augsburg15DetectionDataset, collate_augsburg15_detection
from src.image_tools.overlap import clean_pseudo_labels
from src.models.exponential_moving_average import ExponentialMovingAverage
from src.models.faster_rcnn import PretrainedEfficientNetV2
from src.training.transforms import Compose, ToTensor, RandomHorizontalFlip, RandomVerticalFlip


class SoftTeacher(pl.LightningModule):
    """
    Adapted SoftTeacher model from https://arxiv.org/abs/2106.09018.
    """

    def __init__(
            self,
            num_classes: int,
            batch_size: int,
            teacher_pseudo_threshold: float,
            student_inference_threshold: float,
            unsupervised_loss_weight: float,
    ):
        super(SoftTeacher, self).__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.batch_size = batch_size
        self.unsupervised_loss_weight = unsupervised_loss_weight

        self.student = PretrainedEfficientNetV2(
            num_classes=num_classes,
            batch_size=batch_size
        )
        # Only use high confidence box predictions for inference.
        self.student.model.roi_heads.score_thresh = student_inference_threshold

        self.teacher = deepcopy(self.student)
        self.teacher.freeze()
        # Only use high confidence additional pseudo boxes.
        self.teacher.model.roi_heads.score_thresh = teacher_pseudo_threshold
        self.teacher.eval()
        # TODO decay should change because student learning slows down https://arxiv.org/pdf/1703.01780.pdf.
        self.exponential_moving_average = ExponentialMovingAverage(self.student, self.teacher, decay=0.99)

        self.validation_mean_average_precision = MeanAveragePrecision(class_metrics=True, compute_on_step=False)
        self.test_mean_average_precision = MeanAveragePrecision(class_metrics=True, compute_on_step=False)

    def on_before_zero_grad(self, optimizer: Optimizer) -> None:
        self.exponential_moving_average.update_teacher()

    def forward(self, x, y=None, teacher_box_predictor=None):

        y_labelled = self.student(x, y, teacher_box_predictor, self.unsupervised_loss_weight)

        # Box jittering: use box regression head of teacher (multiple times with different starting points) and look if
        # it comes to the same result (-> reliable regression, higher weight).
        return y_labelled

    def train(self: 'SoftTeacher', mode: bool = True) -> 'SoftTeacher':
        super().train(mode)
        self.teacher.eval()
        return self

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, targets = batch

        # TODO Depends on consistency regularization: ideally, there should be a weakly augmented batch (for the
        # TODO teacher) and a strongly augmented batch (for the student) https://arxiv.org/pdf/2001.07685.pdf.
        # Augment teacher images:
        teacher_augmenter = torchvision.transforms.Compose([
            torchvision.transforms.RandomSolarize(threshold=float(torch.rand(1).numpy()), p=0.25),
            torchvision.transforms.RandomApply(
                [torchvision.transforms.ColorJitter(brightness=(0., 1.), contrast=(0., 1.))],
                p=0.25
            ),
            torchvision.transforms.RandomAdjustSharpness(sharpness_factor=float(torch.rand(1)), p=0.25),
        ])
        teacher_images = teacher_augmenter(images)

        # Originally, this would be two different batches, labelled + unlabelled.
        raw_x_pseudo = self.teacher(teacher_images)
        cleaned_y_pseudo = clean_pseudo_labels(raw_x_pseudo, targets)

        loss_dict = self(images, cleaned_y_pseudo, self.teacher.model.roi_heads.box_predictor)

        total_loss = sum(loss for loss in loss_dict.values())
        self.log('train_loss', total_loss, on_step=True, batch_size=self.batch_size)
        return total_loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        images, targets = batch
        predictions = self(images, targets)
        self.validation_mean_average_precision(predictions, targets)

    def on_validation_end(self) -> None:
        metrics = self.validation_mean_average_precision.compute()
        self._log_metrics(metrics)
        self.validation_mean_average_precision.reset()

    def test_step(self, batch, batch_idx):
        images, targets = batch
        predictions = self(images, targets)
        self.test_mean_average_precision(predictions, targets)

    def on_test_end(self) -> None:
        metrics = self.test_mean_average_precision.compute()
        self._log_metrics(metrics)
        self.test_mean_average_precision.reset()

    def _log_metrics(self, mean_average_precision):
        for index, value in enumerate(mean_average_precision['map_per_class']):
            mean_average_precision[f'map_per_class_{index}'] = value
        for index, value in enumerate(mean_average_precision['mar_100_per_class']):
            mean_average_precision[f'mar_100_per_class_{index}'] = value
        del mean_average_precision['map_per_class']
        del mean_average_precision['mar_100_per_class']
        self.logger.log_metrics(mean_average_precision, step=self.global_step)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.0001)
        return optimizer

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_dataset = Augsburg15DetectionDataset(
            root_directory=os.path.join(os.path.dirname(__file__), '../../datasets/pollen_only'),
            image_info_csv='pollen15_train_annotations_preprocessed.csv',
            transforms=Compose([ToTensor(), RandomHorizontalFlip(0.5), RandomVerticalFlip(0.5)])
        )
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_augsburg15_detection,
            drop_last=True,
            shuffle=True,
            num_workers=4,
            # sampler=WeightedRandomSampler(
            #     weights=train_dataset.get_mean_sample_weights(),
            #     num_samples=100,
            #     replacement=True
            # )
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
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
            num_workers=4
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
            num_workers=4
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass
