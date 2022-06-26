import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader

from src.data_loading.load_augsburg15 import Augsburg15DetectionDataset, collate_augsburg15_detection
from src.evaluation.confusion_matrix import ConfusionMatrix
from src.evaluation.plot_annotated_images import MANUAL_TEST_SET_PATH, MANUAL_ANNOTATIONS_FILE
from src.training.transforms import ToTensor


def get_confusion_matrix(path):
    validation_dataset = Augsburg15DetectionDataset(
        root_directory=MANUAL_TEST_SET_PATH,
        image_info_csv=MANUAL_ANNOTATIONS_FILE,
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

        gt = []
        for box, label in zip(target[0]['boxes'], target[0]['labels']):
            gt.append([label - 1, *box])
        gt = np.array(gt) if len(gt) > 0 else np.array([[]])
        conf_mat.process_batch(preds, gt)

    return conf_mat


def print_confusion_matrix(path):
    matrix = load_confusion_matrix(path)

    print('baseline: ')
    print(matrix)
    print(f'FN: {matrix[:, -1].sum()}')
    print(f'FP: {matrix[-1, :].sum()}')


def save_confusion_matrix(path, output_path):
    matrix = get_confusion_matrix(path).return_matrix()

    with open(output_path, 'wb') as output_file:
        pickle.dump(matrix, output_file)


def load_confusion_matrix(path):
    with open(path, 'rb') as file:
        a = pickle.load(file)
    return a


def visualize_confusion_matrix(path, horizontal_normalization=False, vertical_normalization=False):
    matrix = load_confusion_matrix(path)

    # Sci-kit learn expects ground truth in axis=0
    matrix = matrix.T

    if horizontal_normalization:
        matrix = matrix / np.sum(matrix, axis=1, keepdims=True)
    elif vertical_normalization:
        matrix = matrix / np.sum(matrix, axis=0, keepdims=True)

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
    plt.savefig(path[:-3] + 'png')
    plt.close()


if __name__ == '__main__':
    results = [
        # '/Users/benni/Desktop/ma/evaluation/efficient_net_focal_no_rotation-results.pkl',
        # '/Users/benni/Desktop/ma/evaluation/ssl-focal-no-color-results.pkl',
        # '/Users/benni/Desktop/ma/evaluation/ssl-ce-no-color-results.pkl',
        '/Users/benni/Desktop/ma/evaluation/ssl-focal-no-color-results.pkl.pkl',
    ]
    output = [
        # '/Users/benni/Desktop/ma/evaluation/efficient_net_focal_no_rotation-results.mat',
        # '/Users/benni/Desktop/ma/evaluation/ssl-focal-no-color-results.mat',
        # '/Users/benni/Desktop/ma/evaluation/ssl-ce-no-color-results.mat',
        '/Users/benni/Desktop/ma/evaluation/ssl-focal-no-color-results.mat',
    ]
    # print_confusion_matrix(results[0])
    # visualize_confusion_matrix(results)

    for r, o in zip(results, output):
        # save_confusion_matrix(r, o)
        print_confusion_matrix(o)
        # visualize_confusion_matrix(o, horizontal_normalization=True)
        # print_confusion_matrix(o)
