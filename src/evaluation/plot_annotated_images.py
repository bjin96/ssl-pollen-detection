from typing import Type

import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule

from src.models.soft_teacher import SoftTeacher
from src.training.transforms import ToTensor
import os
from src.data_loading.load_augsburg15 import Augsburg15DetectionDataset, collate_augsburg15_detection
import matplotlib.pyplot as plt
from matplotlib import patches

# Use boxes with higher confidence only.
SCORE_THRESHOLD = 0.5


def plot_annotated_images(checkpoint_path: str, model_class: Type[LightningModule]):
    validation_dataset = Augsburg15DetectionDataset(
        root_directory=os.path.join('../../datasets/pollen_only'),
        image_info_csv='pollen15_val_annotations_preprocessed.csv',
        transforms=ToTensor()
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        collate_fn=collate_augsburg15_detection,
        drop_last=True,
        num_workers=4
    )

    model = model_class.load_from_checkpoint(checkpoint_path, num_classes=Augsburg15DetectionDataset.NUM_CLASSES, batch_size=8)
    model.eval()

    for index, sample in enumerate(validation_loader):
        image, target = sample

        result = model(image)

        plot_bounding_box_image(
            image[0].detach().numpy(),
            result[0]['boxes'].detach().numpy(),
            result[0]['labels'].detach().numpy(),
            result[0]['scores'].detach().numpy(),
            target[0]['boxes'].detach().numpy(),
            target[0]['labels'].detach().numpy(),
            index
        )


def plot_bounding_box_image(image, bounding_boxes, labels, scores, ground_truth_boxes, ground_truth_labels, index):
    plt.rcParams["figure.figsize"] = (20, 20)
    fig, ax = plt.subplots()

    ax.imshow(image.transpose(1, 2, 0))

    _plot_bounding_boxes(bounding_boxes, labels, scores, ax, 'green')
    _plot_bounding_boxes(ground_truth_boxes, ground_truth_labels, np.ones_like(ground_truth_labels), ax, 'red')

    plt.savefig(f'../../plots/{index}.jpg')
    plt.close()


def _plot_bounding_boxes(bounding_boxes, labels, scores, ax, color):
    for bounding_box, label, score in zip(bounding_boxes, labels, scores):
        if score < SCORE_THRESHOLD:
            continue

        width = bounding_box[2] - bounding_box[0]
        height = bounding_box[3] - bounding_box[1]

        rectangle = patches.Rectangle(
            (bounding_box[0], bounding_box[1]),
            width,
            height,
            linewidth=1,
            edgecolor=color,
            facecolor='none'
        )
        label_y = bounding_box[1]
        if color == 'red':
            label_y += height
        ax.add_patch(rectangle)
        ax.text(
            bounding_box[0],
            label_y,
            Augsburg15DetectionDataset.INVERSE_CLASS_MAPPING[int(label)],
            color='white',
            bbox=dict(facecolor=color, edgecolor=color)
        )


if __name__ == '__main__':
    CKPT_PATH = '../../lightning_logs/soft_teacher_#74301e48/checkpoints/epoch=3-step=5907.ckpt'
    plot_annotated_images(CKPT_PATH, SoftTeacher)
