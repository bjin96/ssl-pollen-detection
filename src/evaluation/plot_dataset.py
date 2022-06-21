from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from torch.utils.data import DataLoader

from src.data_loading.load_augsburg15 import Augsburg15DetectionDataset, collate_augsburg15_detection
from src.evaluation.plot_annotated_images import AUGSBURG15_2016_2018_PATH, ANNOTATIONS_FILE, AUGSBURG15_2016_PATH, \
    AUGSBURG15_2016_TEST_FILE, AUGSBURG15_2016_VALIDATION_FILE, AUGSBURG15_2016_TRAIN_FILE, MANUAL_TEST_SET_PATH, \
    MANUAL_ANNOTATIONS_FILE
from src.training.transforms import ToTensor, RandomHorizontalFlip, RandomVerticalFlip, RandomCrop, RandomRotation, \
    Compose


def plot_frequency_per_time(dataset_info, output_name, label_key, file_path_key, month_interval, log_scale):
    plt.rcParams["figure.figsize"] = (10, 8)

    dataset_info['date'] = pd.to_datetime(dataset_info[file_path_key].str[:10], format='%Y%m%d%H')

    frequencies_per_time_and_class = []

    for class_name in Augsburg15DetectionDataset.CLASS_MAPPING.keys():
        frequencies_per_time_and_class.append(get_per_month_per_class(dataset_info, class_name, label_key))
    frequencies_per_time_and_class = pd.concat(frequencies_per_time_and_class, axis=1).fillna(0)

    x = frequencies_per_time_and_class.index.tolist()
    x = [datetime(d.year, d.month, 15).date() for d in x]

    fig, ax = plt.subplots()

    xfmt = mdates.DateFormatter("%b '%y")
    ax.xaxis.set_major_formatter(xfmt)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=month_interval))

    previous = None
    for index, (name, column) in enumerate(frequencies_per_time_and_class.iteritems()):
        current = column.to_numpy()
        if log_scale:
            current = np.where(current > 0, np.log2(current), 0)
        current = current.astype(int)
        ax.bar(x, current, width=15, label=name, bottom=previous, color=plt.get_cmap('tab20').colors[index])
        if previous is None:
            previous = current
        else:
            previous += current
    ax.legend()
    plt.savefig(
        f'../../plots/dataset_exploration/{output_name}{"_log" if log_scale else ""}.pdf',
        bbox_inches='tight')
    plt.close()
    # plt.show()


def get_per_month_per_class(dataset_info, class_name, label_key):
    class_info = dataset_info.loc[dataset_info[label_key] == class_name]
    grouped = class_info.groupby(pd.Grouper(key='date', freq='M')).count()

    grouped[class_name] = grouped[label_key]

    return grouped[[class_name]]


def get_augsburg15_2016_info():
    infos = [
        pd.read_csv(f'{AUGSBURG15_2016_PATH}/{AUGSBURG15_2016_TRAIN_FILE}', header=None),
        pd.read_csv(f'{AUGSBURG15_2016_PATH}/{AUGSBURG15_2016_VALIDATION_FILE}', header=None),
        pd.read_csv(f'{AUGSBURG15_2016_PATH}/{AUGSBURG15_2016_TEST_FILE}', header=None)
    ]
    return pd.concat(infos, axis=0)


def get_augsburg15_2016_2018_info():
    return pd.read_csv(f'{AUGSBURG15_2016_2018_PATH}/{ANNOTATIONS_FILE}')


def get_manual_test_set_info():
    return pd.read_csv(f'{MANUAL_TEST_SET_PATH}/{MANUAL_ANNOTATIONS_FILE}')


def create_time_augsburg15_2016(log_scale):
    info = get_augsburg15_2016_info()
    plot_frequency_per_time(
        info,
        'time_augsburg15_2016',
        label_key=5,
        file_path_key=0,
        month_interval=3,
        log_scale=log_scale
    )


def create_time_augsburg15_2016_2018(log_scale):
    info = get_augsburg15_2016_2018_info()
    plot_frequency_per_time(
        info,
        'time_augsburg15_2016_2018',
        label_key='label',
        file_path_key='file_path',
        month_interval=6,
        log_scale=log_scale
    )


def plot_class_counts(dataset_info, output_name, label_key, file_path_key):
    plt.rcParams["figure.figsize"] = (12, 8)

    class_counts = dataset_info.groupby(label_key).count()

    class_names = np.flip(class_counts.index.to_numpy())
    class_counts = np.flip(class_counts[file_path_key].to_numpy())

    fig, ax = plt.subplots()

    bar_list = plt.barh(class_names, class_counts)
    ax.bar_label(bar_list)

    for index, _ in enumerate(class_names):
        bar_list[index].set_color(plt.get_cmap('tab20').colors[len(class_names) - index - 1])

    plt.savefig(
        f'../../plots/dataset_exploration/{output_name}.pdf',
        bbox_inches='tight')
    plt.close()
    # plt.show()


def create_class_counts_augsburg15_2016_2018():
    info = get_augsburg15_2016_2018_info()
    plot_class_counts(
        info,
        'class_counts_augsburg15_2016_2018',
        label_key='label',
        file_path_key='file_path',
    )


def create_class_counts_augsburg15_2016():
    info = get_augsburg15_2016_info()
    plot_class_counts(
        info,
        'class_counts_augsburg15_2016',
        label_key=5,
        file_path_key=0,
    )


def plot_heatmap_box_locations(dataset_info, output_name, x1_key, y1_key, x2_key, y2_key):
    plt.rcParams["figure.figsize"] = (12, 8)

    heatmap = np.zeros((960, 1280))

    fig, ax = plt.subplots()

    for index, row in dataset_info.iterrows():
        heatmap[row[y1_key]:row[y2_key], row[x1_key]:row[x2_key]] += 1

    image = ax.imshow(heatmap, cmap='viridis')
    fig.colorbar(image)

    plt.savefig(
        f'../../plots/dataset_exploration/{output_name}.pdf',
        bbox_inches='tight')
    plt.close()
    # plt.show()


def create_heatmap_box_location_augsburg15_2016():
    info = get_augsburg15_2016_info()
    plot_heatmap_box_locations(
        info,
        'heatmap_box_locations_augsburg15_2016',
        x1_key=1,
        y1_key=2,
        x2_key=3,
        y2_key=4,
    )


def create_heatmap_box_location_augmented_augsburg15_2016():
    validation_dataset = Augsburg15DetectionDataset(
        root_directory=AUGSBURG15_2016_PATH,
        image_info_csv=AUGSBURG15_2016_TRAIN_FILE,
        transforms=Compose([
            ToTensor(),
            RandomHorizontalFlip(0.5),
            RandomVerticalFlip(0.5),
            RandomCrop(0.5),
            RandomRotation(0.5, 25, (1280, 960)),
        ])
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        collate_fn=collate_augsburg15_detection,
        drop_last=True,
        num_workers=4
    )
    info = []
    for index, (image, target) in enumerate(validation_loader):
        info.extend(target[0]['boxes'].detach().numpy().astype(int).tolist())
    info = pd.DataFrame(info)
    plot_heatmap_box_locations(
        info,
        'heatmap_box_locations_augsburg15_2016_augmented',
        x1_key=0,
        y1_key=1,
        x2_key=2,
        y2_key=3,
    )


def create_heatmap_box_location_augsburg15_2016_2018():
    info = get_augsburg15_2016_2018_info()
    plot_heatmap_box_locations(
        info,
        'heatmap_box_locations_augsburg15_2016_2018',
        x1_key='x1',
        y1_key='y1',
        x2_key='x2',
        y2_key='y2',
    )


def create_heatmap_box_location_manual_test_set():
    info = get_manual_test_set_info()
    plot_heatmap_box_locations(
        info,
        'heatmap_box_locations_manual_test_set',
        x1_key='x1',
        y1_key='y1',
        x2_key='x2',
        y2_key='y2',
    )


if __name__ == '__main__':
    # create_time_augsburg15_2016(False)
    # create_time_augsburg15_2016_2018(False)

    # create_class_counts_augsburg15_2016_2018()
    # create_class_counts_augsburg15_2016()

    create_heatmap_box_location_augsburg15_2016()
    create_heatmap_box_location_augsburg15_2016_2018()
    create_heatmap_box_location_manual_test_set()
    #
    # create_heatmap_box_location_augmented_augsburg15_2016()
