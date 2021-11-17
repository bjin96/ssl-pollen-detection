import os
from abc import abstractmethod

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader
from torchmetrics import MAP

from data_loading.load_augsburg15 import Augsburg15DetectionDataset, collate_augsburg15_detection
from training.transforms import Compose, ToTensor, RandomHorizontalFlip


class ObjectDetector(LightningModule):

    def __init__(self, num_classes, batch_size):
        super().__init__()
        self.num_classes = num_classes
        self.model = self.define_model()
        self.mean_average_precision = MAP()
        self.batch_size = batch_size

    @abstractmethod
    def define_model(self):
        pass

    def forward(self, x):
        y = self.model(x)
        return y

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, targets = batch

        loss_dict = self.model(images, targets)

        total_loss = sum(loss for loss in loss_dict.values())
        self.log('train_loss', total_loss, on_step=True, batch_size=self.batch_size)
        return total_loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        predictions = self.model(images, targets)

        mean_average_precision = self.mean_average_precision(predictions, targets)
        self.log('map@0.50:0.95', mean_average_precision['map'], on_epoch=True, batch_size=self.batch_size)
        self.log('map@0.50', mean_average_precision['map_50'], on_epoch=True, batch_size=self.batch_size)
        self.log('map@0.75', mean_average_precision['map_75'], on_epoch=True, batch_size=self.batch_size)
        self.log('mar@100', mean_average_precision['mar_100'], on_epoch=True, batch_size=self.batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_dataset = Augsburg15DetectionDataset(
            root_directory=os.path.join(os.path.dirname(__file__), '../datasets/pollen_only'),
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
        pass

    def val_dataloader(self) -> EVAL_DATALOADERS:
        validation_dataset = Augsburg15DetectionDataset(
            root_directory=os.path.join(os.path.dirname(__file__), '../datasets/pollen_only'),
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
