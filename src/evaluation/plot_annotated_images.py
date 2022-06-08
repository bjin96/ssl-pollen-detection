import pickle
from typing import Type

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule

from src.evaluation.confusion_matrix import ConfusionMatrix
from src.models.soft_teacher import SoftTeacher
from src.training.transforms import ToTensor
import os
from src.data_loading.load_augsburg15 import Augsburg15DetectionDataset, collate_augsburg15_detection
import matplotlib.pyplot as plt
from matplotlib import patches


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


def plot_ground_truth_images():
    validation_dataset = Augsburg15DetectionDataset(
        root_directory=os.path.join('/Volumes/Benni T5/master_data/2018'),
        image_info_csv='manual_annotations.csv',
        transforms=ToTensor()
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        collate_fn=collate_augsburg15_detection,
        drop_last=True,
        num_workers=4
    )

    for index, sample in enumerate(validation_loader):
        image, target = sample

        plot_bounding_box_image(
            image[0].detach().numpy(),
            [],
            [],
            [],
            target[0]['boxes'].detach().numpy(),
            target[0]['labels'],
            target[0]['updated'].detach().numpy(),
            index
        )


def plot_from_results(path_a):
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

    with open(path_a, 'rb') as file:
        a = pickle.load(file)

    for index, sample in enumerate(validation_loader):
        image, target = sample

        plot_bounding_box_image_scores(
            image[0].detach().numpy(),
            a[index][0].numpy(),
            a[index][1].numpy(),
            a[index][2].numpy(),
            target[0]['boxes'].detach().numpy(),
            target[0]['labels'].detach().numpy(),
            np.array([1.0 for _ in target[0]['boxes'].detach()]),
            index
        )


def make_validation_predictions(checkpoint_path, model_class):
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

    model = model_class.load_from_checkpoint(
        checkpoint_path,
        num_classes=Augsburg15DetectionDataset.NUM_CLASSES,
        batch_size=1
    )
    model.eval()

    mps_device = torch.device('mps')
    model.to(mps_device)

    results = []

    for index, sample in enumerate(validation_loader):
        print(f'{index + 1}/{len(validation_dataset)}')
        image, target = sample

        image = image.to(mps_device)

        result = model(image)

        results.append((result[0]['boxes'].detach(), result[0]['labels'].detach(), result[0]['scores'].detach()))

    with open('results.pkl', 'w') as file:
        pickle.dump(results, file)


def plot_bounding_box_image_scores(image, bounding_boxes, labels, scores, ground_truth_boxes, ground_truth_labels, ground_truth_scores, index):
    scores_mask = scores > 0.5
    ground_truth_scores_mask = ground_truth_scores > 0.5

    bounding_boxes = bounding_boxes[scores_mask]
    labels = labels[scores_mask]

    ground_truth_boxes = ground_truth_boxes[ground_truth_scores_mask]
    ground_truth_labels = ground_truth_labels[ground_truth_scores_mask]

    if not is_interesting_image(bounding_boxes, labels, ground_truth_boxes, ground_truth_labels):
        return

    plt.rcParams["figure.figsize"] = (20, 20)
    fig, ax = plt.subplots()

    ax.imshow(image.transpose(1, 2, 0))

    _plot_bounding_boxes(bounding_boxes, labels, scores, ax, 'green')
    _plot_bounding_boxes(ground_truth_boxes, ground_truth_labels, ground_truth_scores, ax, 'red')

    plt.savefig(f'../../plots/interesting_ssl_ce_no_color/{index}.jpg', bbox_inches='tight')
    plt.close()


def plot_bounding_box_image(image, bounding_boxes, labels, scores, ground_truth_boxes, ground_truth_labels, updated, index):
    plt.rcParams["figure.figsize"] = (20, 20)
    fig, ax = plt.subplots()

    ax.imshow(image.transpose(1, 2, 0))

    _plot_bounding_boxes(bounding_boxes, labels, scores, ax, 'green', [])
    _plot_bounding_boxes(ground_truth_boxes, ground_truth_labels, np.ones_like(ground_truth_labels), ax, 'red', updated)

    plt.savefig(f'../../plots/manual_labels/{index}.jpg', bbox_inches='tight')
    plt.close()


def is_interesting_image(a_boxes, a_labels, b_boxes, b_labels):
    for box, label in zip(a_boxes, a_labels):
        if is_interesting_box(box, label, b_boxes, b_labels):
            return True

    for box, label in zip(b_boxes, b_labels):
        if is_interesting_box(box, label, a_boxes, a_labels):
            return True

    return False


def is_interesting_box(box, label, other_boxes, other_labels):
    for other_box, other_label in zip(other_boxes, other_labels):
        vertical_start_inside = other_box[0] <= box[0] <= other_box[2]
        vertical_end_inside = other_box[0] <= box[2] <= other_box[2]
        vertical_enclosing = box[0] <= other_box[0] <= box[2]
        vertical_overlap = vertical_start_inside or vertical_end_inside or vertical_enclosing

        horizontal_start_inside = other_box[1] <= box[1] <= other_box[3]
        horizontal_end_inside = other_box[1] <= box[3] <= other_box[3]
        horizontal_enclosing = box[1] <= other_box[1] <= box[3]
        horizontal_overlap = horizontal_start_inside or horizontal_end_inside or horizontal_enclosing

        if vertical_overlap and horizontal_overlap:
            if label == other_label:
                return False

    return True


def _plot_bounding_boxes(bounding_boxes, labels, scores, ax, color, updated):
    for bounding_box, label, score, u in zip(bounding_boxes, labels, scores, updated):
        width = bounding_box[2] - bounding_box[0]
        height = bounding_box[3] - bounding_box[1]

        if u:
            color = 'green'

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
            f'{label} {score}',
            color='white',
            bbox=dict(facecolor=color, edgecolor=color)
        )


def count_bounding_boxes(path, p):
    with open(path, 'rb') as file:
        a = pickle.load(file)

    count = 0

    for boxes, labels, scores in a:
        count += len([score for score in scores if score > p])

    return count


def plot_image_with_bounding_boxes(image_path, output_path, bounding_boxes, labels):
    plt.rcParams["figure.figsize"] = (20, 20)
    fig, ax = plt.subplots()

    image = np.asarray(Image.open(image_path))

    ax.imshow(image, cmap='gray', vmin=0, vmax=255)

    _plot_bounding_boxes(bounding_boxes, labels, ['' for _ in labels], ax, 'green')

    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def crop_bounding_box(image_path, bounding_box, output_path):
    image = Image.open(image_path).crop(bounding_box)
    image.save(output_path)


if __name__ == '__main__':
    # ssl_results = '/Users/benni/Desktop/ma/evaluation/ssl-results.pkl'
    # baseline_results = '/Users/benni/Desktop/ma/evaluation/baseline-results.pkl'
    # ssl_fixed_ema = '/Users/benni/Desktop/ma/evaluation/ssl-fixed-ema-results.pkl'

    # results = [
    #     '/Users/benni/Desktop/ma/evaluation/efficient_net_focal_no_rotation-results.pkl',
    #     '/Users/benni/Desktop/ma/evaluation/ssl-focal-no-color-results.pkl',
    #     '/Users/benni/Desktop/ma/evaluation/ssl-ce-no-color-results.pkl',
    # ]

    # CKPT_PATH = '/Users/benni/Desktop/ma/cluster_results/ssl-fixed-ema/epoch=11-step=35460.ckpt'
    # plot_annotated_images(CKPT_PATH, SoftTeacher)
    # make_validation_predictions(CKPT_PATH, SoftTeacher)
    # plot_from_results(results[2])

    # print(count_bounding_boxes(baseline_results, 0.0))

    # print_confusion_matrix(ssl_fixed_ema)
    plot_ground_truth_images()

    # plot_image_with_bounding_boxes(
    #     '/Volumes/Benni T5/master_data/2018/20180411180809_A034779/images/polle-im_01_13_17-20180411180809-pmon-00013-A034779-tiffFAST.SYN._FP.png',
    #     '/Users/benni/Desktop/ma/documents/meeting_22_06_01/vbetula.png',
    #     [[257, 465, 349, 558]],
    #     ['VBetula']
    # )

    # crop_bounding_box(
    #     '/Volumes/Benni T5/master_data/2018/20180411180809_A034779/images/polle-im_01_13_17-20180411180809-pmon-00013-A034779-tiffFAST.SYN._FP.png',
    #     [257, 465, 349, 558],
    #     '/Users/benni/Desktop/ma/documents/meeting_22_06_01/vbetula_bbox.png',
    # )

