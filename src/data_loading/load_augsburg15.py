import os

import pandas as pd
import numpy as np
from torch import FloatTensor, LongTensor, Tensor, ByteTensor, stack
from torch.utils.data import Dataset
from PIL import Image


class Augsburg15DetectionDataset(Dataset):

    CLASS_MAPPING = {
        'Alnus': 1,
        'Betula': 2,
        'Carpinus': 3,
        'Corylus': 4,
        'Fagus': 5,
        'Fraxinus': 6,
        'Plantago': 7,
        'Poaceae': 8,
        'Populus': 9,
        'Quercus': 10,
        'Salix': 11,
        'Taxus': 12,
        'Tilia': 13,
        'Ulmus': 14,
        'Urticaceae': 15
    }
    INVERSE_CLASS_MAPPING = [
        'Background',
        'Alnus',
        'Betula',
        'Carpinus',
        'Corylus',
        'Fagus',
        'Fraxinus',
        'Plantago',
        'Poaceae',
        'Populus',
        'Quercus',
        'Salix',
        'Taxus',
        'Tilia',
        'Ulmus',
        'Urticaceae'
    ]
    CLASS_WEIGHTS = [
        # background class at index 0
        1.,
        7016 / 6028,
        7016 / 1465,
        7016 / 4806,
        7016 / 7016,
        7016 / 457,
        7016 / 282,
        7016 / 1016,
        7016 / 2121,
        7016 / 1208,
        7016 / 349,
        7016 / 309,
        7016 / 3691,
        7016 / 109,
        7016 / 216,
        7016 / 1712,
    ]
    NUM_CLASSES = 16

    def __init__(self, root_directory, image_info_csv, transforms=None):
        self.transforms = transforms
        self.root_directory = root_directory
        self.image_info_csv = image_info_csv

        self.image_info = self._parse_image_info_csv()

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        image_file = os.path.join(self.root_directory, self.image_info['file'][idx])
        image = Image.open(image_file).convert('RGB')
        target = {
            'boxes': FloatTensor(np.array(self.image_info['bounding_boxes'][idx])),
            'labels': LongTensor(np.array(self.image_info['labels'][idx])),
            'image_id': LongTensor([idx]),
            'area': Tensor(self.image_info['area'][idx]),
            'iscrowd': ByteTensor(self.image_info['iscrowd'][idx])
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return [image, target]

    def get_height_and_width(self, idx):
        return self.image_info['height'][idx], self.image_info['width']['idx']

    def get_mean_sample_weights(self):
        sample_weights = []
        for class_indices in self.image_info['labels']:
            mean_weight = np.mean([self.CLASS_WEIGHTS[class_index] for class_index in class_indices])
            sample_weights.append(mean_weight)
        return sample_weights

    def _parse_image_info_csv(self):
        image_info = pd.read_csv(os.path.join(self.root_directory, self.image_info_csv), header=None)
        image_info.columns = ['file', 'x1', 'y1', 'x2', 'y2', 'labels']
        image_info['area'] = (image_info['x2'] - image_info['x1']) * (image_info['y2'] - image_info['y1'])
        image_info['iscrowd'] = 0
        image_info['labels'] = image_info['labels'].map(self.CLASS_MAPPING)
        image_info['bounding_boxes'] = list(image_info[['x1', 'y1', 'x2', 'y2']].to_numpy())
        image_info = image_info.drop(['x1', 'y1', 'x2', 'y2'], axis=1)
        image_info = image_info.groupby('file').aggregate(list)
        image_info['height'] = 960
        image_info['width'] = 1280
        image_info = image_info.reset_index()
        return image_info


def collate_augsburg15_detection(batch):
    images = []
    targets = []
    for image, target in batch:
        images.append(image)
        targets.append({
            'boxes': target['boxes'],
            'labels': target['labels'],
            'image_id': target['image_id'],
            'area': target['area'],
            'iscrowd': target['iscrowd']
        })
    return [stack(images), targets]
