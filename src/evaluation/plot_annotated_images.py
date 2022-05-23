import pickle
from typing import Type

import numpy as np
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


def plot_from_results(path_a, path_b):
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

    with open(path_b, 'rb') as file:
        b = pickle.load(file)

    for index, sample in enumerate(validation_loader):
        image, target = sample

        plot_bounding_box_image_scores(
            image[0].detach().numpy(),
            a[index][0].numpy(),
            a[index][1].numpy(),
            a[index][2].numpy(),
            b[index][0].numpy(),
            b[index][1].numpy(),
            b[index][2].numpy(),
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

    results = []

    for index, sample in enumerate(validation_loader):
        print(f'{index + 1}/{len(validation_dataset)}')
        image, target = sample

        result = model(image)

        results.append((result[0]['boxes'], result[0]['labels'], result[0]['scores']))

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

    plt.savefig(f'../../plots/{index}.jpg')
    plt.close()


def plot_bounding_box_image(image, bounding_boxes, labels, scores, ground_truth_boxes, ground_truth_labels, index):
    plt.rcParams["figure.figsize"] = (20, 20)
    fig, ax = plt.subplots()

    ax.imshow(image.transpose(1, 2, 0))

    _plot_bounding_boxes(bounding_boxes, labels, scores, ax, 'green')
    _plot_bounding_boxes(ground_truth_boxes, ground_truth_labels, np.ones_like(ground_truth_labels), ax, 'red')

    plt.savefig(f'../../plots/{index}.jpg')
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


def _plot_bounding_boxes(bounding_boxes, labels, scores, ax, color):
    for bounding_box, label, score in zip(bounding_boxes, labels, scores):
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
            f'{Augsburg15DetectionDataset.INVERSE_CLASS_MAPPING[int(label)]} {score:.2f}',
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


def get_confusion_matrix(path):
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

    with open(path, 'rb') as file:
        a = pickle.load(file)

    conf_mat = ConfusionMatrix(num_classes=15, CONF_THRESHOLD=0.5, IOU_THRESHOLD=0.5)

    for i, ((_, target), (boxes, labels, scores)) in enumerate(zip(validation_loader, a)):
        print(f'{i}/{len(a)}')
        preds = np.array([[*box, score, label - 1] for box, label, score in zip(boxes, labels, scores)])
        gt = np.array([[label - 1, *box] for box, label in zip(target[0]['boxes'], target[0]['labels'])])
        conf_mat.process_batch(preds, gt)

    return conf_mat


def compare_confusion_matrix(path_a, path_b):
    matrix_baseline = get_confusion_matrix(path_a)
    matrix_ssl = get_confusion_matrix(path_b)

    print('baseline: ')
    matrix_baseline.print_matrix()
    print(f'FP: {matrix_baseline.return_matrix()[:, -1].sum()}')
    print(f'FN: {matrix_baseline.return_matrix()[-1, :].sum()}')

    print('ssl:')
    matrix_ssl.print_matrix()
    print(f'FP: {matrix_ssl.return_matrix()[:, -1].sum()}')
    print(f'FN: {matrix_ssl.return_matrix()[-1, :].sum()}')

    print()


if __name__ == '__main__':
    ssl_results = '/Users/benni/Desktop/ma/evaluation/ssl-results.pkl'
    baseline_results = '/Users/benni/Desktop/ma/evaluation/baseline-results.pkl'

    # CKPT_PATH = '/Users/benni/Desktop/ma/cluster_results/ssl-baseline/epoch=9-step=29550.ckpt'
    # plot_annotated_images(CKPT_PATH, SoftTeacher)
    # make_validation_predictions(CKPT_PATH, SoftTeacher)
    plot_from_results(ssl_results, baseline_results)

    # print(count_bounding_boxes(baseline_results, 0.0))

    # compare_confusion_matrix(baseline_results, ssl_results)
