import os
from typing import Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

import model_definition
from training.engine import train_one_epoch, evaluate
from torch.utils.data import DataLoader
from model_definition.faster_rcnn import FastRCNNPredictor

from data_loading.load_augsburg15 import Augsburg15DetectionDataset, collate_augsburg15_detection
from training.transforms import ToTensor, RandomHorizontalFlip, Compose


def get_fasterrcnn_model():
    model = model_definition.faster_rcnn.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, Augsburg15DetectionDataset.NUM_CLASSES)
    return model


def fine_tune_faster_rcnn(num_epochs, batch_size=4, print_frequency=10):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_dataset = Augsburg15DetectionDataset(
        root_directory=os.path.join(os.path.dirname(__file__), 'datasets/pollen_only'),
        image_info_csv='pollen15_train_annotations_preprocessed.csv',
        transforms=Compose([ToTensor(), RandomHorizontalFlip(0.5)])
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
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

    validation_dataset = Augsburg15DetectionDataset(
        root_directory=os.path.join(os.path.dirname(__file__), 'datasets/pollen_only'),
        image_info_csv='pollen15_val_annotations_preprocessed.csv',
        transforms=ToTensor()
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        collate_fn=collate_augsburg15_detection,
        drop_last=True,
        num_workers=4
    )

    model = get_fasterrcnn_model()
    model.to(device)

    # Train classifier only:
    # params = model.roi_heads.box_predictor.cls_score.parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.0001)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)

    model.load_state_dict(torch.load('model_epoch_11'))

    for epoch in range(num_epochs):
        metric_logger = train_one_epoch(model, optimizer, train_loader, device, epoch, print_frequency)
        torch.save(model.state_dict(), f'models/model_epoch_{epoch}')
        # lr_scheduler.step()
        _, metric_logger = evaluate(model, validation_loader, device)
        print(metric_logger.meters['loss'].value)


if __name__ == '__main__':
    fine_tune_faster_rcnn(40, 16)
