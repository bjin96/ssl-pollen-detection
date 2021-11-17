import torch
from torch.utils.data import DataLoader

from fine_tune_faster_rcnn import get_fasterrcnn_model
from training.transforms import ToTensor
import os
from data_loading.load_augsburg15 import Augsburg15DetectionDataset, collate_augsburg15_detection
import matplotlib.pyplot as plt
from matplotlib import patches


def plot_annotated_images(saved_model_path):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    validation_dataset = Augsburg15DetectionDataset(
        root_directory=os.path.join('../datasets/pollen_only'),
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

    model = get_fasterrcnn_model()
    model.load_state_dict(torch.load(saved_model_path))
    model.to(device)
    model.eval()

    for index, sample in enumerate(validation_loader):
        image, target = sample
        image = image.to(device)

        result = model(image)

        plot_bounding_box_image(
            image[0],
            result[0]['boxes'],
            result[0]['labels'],
            target[0]['boxes'],
            target[0]['labels'],
            index
        )


def plot_bounding_box_image(image, bounding_boxes, labels, ground_truth_boxes, ground_truth_labels, index):
    plt.rcParams["figure.figsize"] = (20, 20)
    fig, ax = plt.subplots()

    ax.imshow(image.cpu().permute(1, 2, 0))

    _plot_bounding_boxes(bounding_boxes, labels, ax, 'green')
    _plot_bounding_boxes(ground_truth_boxes, ground_truth_labels, ax, 'red')

    plt.savefig(f'./plots/{index}.jpg')
    plt.close()


def _plot_bounding_boxes(bounding_boxes, labels, ax, color):
    for i, bounding_box in enumerate(bounding_boxes):
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
            Augsburg15DetectionDataset.INVERSE_CLASS_MAPPING[int(labels[i])],
            color='white',
            bbox=dict(facecolor=color, edgecolor=color)
        )


if __name__ == '__main__':
    plot_annotated_images('../models/model_epoch_25')
