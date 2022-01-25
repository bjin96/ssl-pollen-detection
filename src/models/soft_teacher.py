import os
from copy import deepcopy
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS, STEP_OUTPUT
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader
from torchmetrics.detection.map import MeanAveragePrecision

from src.data_loading.load_augsburg15 import Augsburg15DetectionDataset, collate_augsburg15_detection
from src.image_tools.overlap import clean_pseudo_labels
from src.models.exponential_moving_average import ExponentialMovingAverage
from src.models.faster_rcnn import PretrainedEfficientNetV2
from src.training.transforms import Compose, ToTensor, RandomHorizontalFlip


class SoftTeacher(pl.LightningModule):
    """
    Adapted SoftTeacher model from https://arxiv.org/abs/2106.09018.
    """

    def __init__(self, num_classes, batch_size):
        super(SoftTeacher, self).__init__()

        self.num_classes = num_classes
        self.batch_size = batch_size

        self.student = PretrainedEfficientNetV2(
            num_classes=num_classes,
            batch_size=batch_size
        )
        self.teacher = deepcopy(self.student)
        self.teacher.freeze()
        # Only use high confidence additional pseudo boxes.
        self.teacher.model.roi_heads.score_thresh = 0.9
        self.teacher.eval()
        # TODO decay should change because student learning slows down https://arxiv.org/pdf/1703.01780.pdf.
        self.exponential_moving_average = ExponentialMovingAverage(self.student, self.teacher, decay=0.99)

        self.validation_mean_average_precision = MeanAveragePrecision(class_metrics=True, compute_on_step=False)
        self.test_mean_average_precision = MeanAveragePrecision(class_metrics=True, compute_on_step=False)

    def on_before_zero_grad(self, optimizer: Optimizer) -> None:
        self.exponential_moving_average.update_teacher()

    def forward(self, x, y=None, teacher_box_predictor=None):

        y_labelled = self.student(x, y, teacher_box_predictor)

        # TODO Weighting supervised + unsupervised loss: now both are handled equally.

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
        # TODO teacher) and a strongly augmented batch (for the student).

        # Originally, this would be two different batches, labelled + unlabelled.
        raw_x_pseudo = self.teacher(images)
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
        self.log('map@0.50:0.95', mean_average_precision['map'], on_epoch=True, batch_size=self.batch_size)
        self.log('map@0.50', mean_average_precision['map_50'], on_epoch=True, batch_size=self.batch_size)
        self.log('map@0.75', mean_average_precision['map_75'], on_epoch=True, batch_size=self.batch_size)
        self.log('map_small', mean_average_precision['map_small'], on_epoch=True, batch_size=self.batch_size)
        self.log('map_medium', mean_average_precision['map_medium'], on_epoch=True, batch_size=self.batch_size)
        self.log('map_large', mean_average_precision['map_large'], on_epoch=True, batch_size=self.batch_size)
        self.log('mar@1', mean_average_precision['mar_1'], on_epoch=True, batch_size=self.batch_size)
        self.log('mar@10', mean_average_precision['mar_10'], on_epoch=True, batch_size=self.batch_size)
        self.log('mar@100', mean_average_precision['mar_100'], on_epoch=True, batch_size=self.batch_size)
        self.log('mar_small', mean_average_precision['mar_small'], on_epoch=True, batch_size=self.batch_size)
        self.log('mar_medium', mean_average_precision['mar_medium'], on_epoch=True, batch_size=self.batch_size)
        self.log('mar_large', mean_average_precision['mar_large'], on_epoch=True, batch_size=self.batch_size)

        for index, label in enumerate(Augsburg15DetectionDataset.INVERSE_CLASS_MAPPING):
            self.log(
                f'map_{label}', mean_average_precision['map_per_class'][index],
                on_epoch=True,
                batch_size=self.batch_size
            )
            self.log(
                f'mar_100_{label}', mean_average_precision['mar_100_per_class'][index],
                on_epoch=True,
                batch_size=self.batch_siz
            )

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.0001)
        return optimizer

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_dataset = Augsburg15DetectionDataset(
            root_directory=os.path.join(os.path.dirname(__file__), '../../datasets/pollen_only'),
            image_info_csv='pollen15_train_annotations_preprocessed.csv',
            transforms=Compose([ToTensor(), RandomHorizontalFlip(0.5)])
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
