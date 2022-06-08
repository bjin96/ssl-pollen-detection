import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader

from src.data_loading.load_augsburg15 import Augsburg15DetectionDataset, collate_augsburg15_detection
from src.evaluation.confusion_matrix import ConfusionMatrix
from src.training.transforms import ToTensor


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

    conf_mat = ConfusionMatrix(num_classes=15, CONF_THRESHOLD=0.9, IOU_THRESHOLD=0.5)

    for i, ((_, target), (boxes, labels, scores)) in enumerate(zip(validation_loader, a)):
        print(f'{i}/{len(a)}')
        preds = np.array([[*box, score, label - 1] for box, label, score in zip(boxes, labels, scores)])
        gt = np.array([[label - 1, *box] for box, label in zip(target[0]['boxes'], target[0]['labels'])])
        conf_mat.process_batch(preds, gt)

    return conf_mat


def print_confusion_matrix(path):
    matrix = load_confusion_matrix(path)

    print('baseline: ')
    print(matrix)
    print(f'FP: {matrix[:, -1].sum()}')
    print(f'FN: {matrix[-1, :].sum()}')


def save_confusion_matrix(path, output_path):
    matrix = get_confusion_matrix(path).return_matrix()

    with open(output_path, 'wb') as output_file:
        pickle.dump(matrix, output_file)


def load_confusion_matrix(path):
    with open(path, 'rb') as file:
        a = pickle.load(file)
    return a


def visualize_confusion_matrix(path):
    matrix = load_confusion_matrix(path)

    matrix = matrix / np.sum(matrix, axis=1, keepdims=True)

    plt.rcParams["figure.figsize"] = (10, 10)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=Augsburg15DetectionDataset.INVERSE_CLASS_MAPPING[1:] + ['Background']
    )
    disp.plot(
        xticks_rotation='vertical',
        values_format='.2f',
        colorbar=False
    )
    plt.savefig(path[:-3] + '.png')
    plt.close()


if __name__ == '__main__':
    results = [
        '/Users/benni/Desktop/ma/evaluation/efficient_net_focal_no_rotation-results.pkl',
        '/Users/benni/Desktop/ma/evaluation/ssl-focal-no-color-results.pkl',
        '/Users/benni/Desktop/ma/evaluation/ssl-ce-no-color-results.pkl',
    ]
    output = [
        '/Users/benni/Desktop/ma/evaluation/efficient_net_focal_no_rotation-results.mat',
        '/Users/benni/Desktop/ma/evaluation/ssl-focal-no-color-results.mat',
        '/Users/benni/Desktop/ma/evaluation/ssl-ce-no-color-results.mat',
    ]
    # print_confusion_matrix(results[0])
    # visualize_confusion_matrix(results)

    for r, o in zip(results, output):
        # save_confusion_matrix(r, o)
        visualize_confusion_matrix(o)
        # print_confusion_matrix(o)
