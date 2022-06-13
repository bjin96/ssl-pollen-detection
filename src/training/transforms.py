"""Copied from https://github.com/pytorch/vision/tree/main/references/detection"""
import random
from typing import Tuple

import torch
from torch import Tensor
from torchvision.transforms import functional as F


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-2)
            bbox = target["boxes"]
            bbox[:, [1, 3]] = height - bbox[:, [3, 1]]
            target["boxes"] = bbox
        return image, target


class RandomRotation(object):
    def __init__(self, prob, degree_range, image_size, remove_cut_off_images=False, min_area=100):
        self.prob = prob
        self.degree_range = degree_range
        self.image_size = image_size
        self.remove_cut_off_images = remove_cut_off_images
        self.min_area = min_area

    def __call__(self, image, target):
        if random.random() < self.prob:
            degree = float(torch.rand(1)) * 2 * self.degree_range - self.degree_range
            image = F.rotate(image, angle=degree)
            target['boxes'], inside_mask = self._rotate_coordinates(
                box_coordinates=target['boxes'],
                degree=degree
            )
            target['boxes'] = target['boxes'][inside_mask]
            target['labels'] = target['labels'][inside_mask]
            target['area'] = target['area'][inside_mask]
            target['iscrowd'] = target['iscrowd'][inside_mask]

        return image, target

    def _rotate_coordinates(
            self,
            box_coordinates: Tensor,
            degree: float
    ):
        """
        Rotates bounding box coordinates around the center.

        Rotates the original bounding box and encloses the resulting box with a new box that is parallel to the image
        border. Marks a box if it leaves the image boundaries.

        Args:
            box_coordinates: Coordinates of the form [[x1, y1, x2, y3], ...] where (x1, y1) denotes the upper left
                corner and (x2, y2) denotes the lower right corner of the bounding box.
            degree: Degree for which to rotate the coordinates counter-clockwise.

        Returns:
            box_coordinates: New coordinates for the bounding box after rotation.
            inside_mask: Boolean mask to indicate whether the box is inside the image boundaries.
        """
        # Make rotation counter-clockwise
        degree = torch.tensor(-degree)

        radians = degree * torch.pi / 180

        x_center = self.image_size[0] / 2
        y_center = self.image_size[1] / 2

        x_minus_center = box_coordinates[:, [0, 2, 2, 0]] - x_center
        y_minus_center = box_coordinates[:, [1, 3, 1, 3]] - y_center

        x_rotated = x_minus_center * torch.cos(radians) - y_minus_center * torch.sin(radians) + x_center
        y_rotated = x_minus_center * torch.sin(radians) + y_minus_center * torch.cos(radians) + y_center

        # TODO: Maybe do something about bigger bounding boxes...
        box_coordinates[:, 0] = torch.maximum(torch.min(x_rotated, dim=-1)[0], torch.tensor(0))
        box_coordinates[:, 0] = torch.minimum(torch.min(x_rotated, dim=-1)[0], torch.tensor(self.image_size[0]))
        box_coordinates[:, 2] = torch.maximum(torch.max(x_rotated, dim=-1)[0], torch.tensor(0))
        box_coordinates[:, 2] = torch.minimum(torch.max(x_rotated, dim=-1)[0], torch.tensor(self.image_size[0]))
        box_coordinates[:, 1] = torch.maximum(torch.min(y_rotated, dim=-1)[0], torch.tensor(0))
        box_coordinates[:, 1] = torch.minimum(torch.min(y_rotated, dim=-1)[0], torch.tensor(self.image_size[1]))
        box_coordinates[:, 3] = torch.maximum(torch.max(y_rotated, dim=-1)[0], torch.tensor(0))
        box_coordinates[:, 3] = torch.minimum(torch.max(y_rotated, dim=-1)[0], torch.tensor(self.image_size[1]))

        if self.remove_cut_off_images:
            horizontal_inside_mask = torch.logical_and(
                torch.not_equal(box_coordinates[:, 0], torch.tensor(0)),
                torch.not_equal(box_coordinates[:, 2], torch.tensor(self.image_size[0]))
            )
            vertical_inside_mask = torch.logical_and(
                torch.not_equal(box_coordinates[:, 1], torch.tensor(0)),
                torch.not_equal(box_coordinates[:, 3], torch.tensor(self.image_size[1]))
            )
            inside_mask = torch.logical_and(horizontal_inside_mask, vertical_inside_mask)
        else:
            inside_mask = torch.greater_equal(_calculate_batched_box_area(box_coordinates), self.min_area)

        return box_coordinates, inside_mask


class RandomCrop(object):
    def __init__(
            self,
            prob: float,
            min_scale: float = 0.7,
            max_scale: float = 0.9,
            image_size: Tuple = (1280, 960),
            min_area: int = 100,
    ):
        self.prob = prob
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.image_size = image_size
        self.min_area = min_area

    def __call__(self, image, target):
        if random.random() < self.prob:
            scale = (float(torch.rand(1)) * (self.max_scale - self.min_scale)) + self.min_scale

            new_width = int(round(scale * self.image_size[0]))
            new_height = int(round(scale * self.image_size[1]))

            x_crop_origin = (self.image_size[0] - new_width) // 2
            y_crop_origin = (self.image_size[1] - new_height) // 2

            image = F.resized_crop(
                img=image,
                top=y_crop_origin,
                left=x_crop_origin,
                height=new_height,
                width=new_width,
                size=list(reversed(self.image_size))
            )
            target['boxes'], inside_mask = self._calculate_cropped_coordinates(
                box_coordinates=target['boxes'],
                x_crop_origin=x_crop_origin,
                y_crop_origin=y_crop_origin,
                new_width=new_width,
                new_height=new_height,
            )
            target['boxes'] = target['boxes'][inside_mask]
            target['labels'] = target['labels'][inside_mask]
            target['area'] = target['area'][inside_mask]
            target['iscrowd'] = target['iscrowd'][inside_mask]

        return image, target

    def _calculate_cropped_coordinates(
            self,
            box_coordinates: Tensor,
            x_crop_origin: int,
            y_crop_origin: int,
            new_width: int,
            new_height: int,
    ):
        """
        Calculates new bounding box coordinates after a center crop. Marks a box if it leaves the image boundaries.

        Args:
            box_coordinates: Coordinates of the form [[x1, y1, x2, y3], ...] where (x1, y1) denotes the upper left
                corner and (x2, y2) denotes the lower right corner of the bounding box.
            x_crop_origin: New origin after cropping.
            y_crop_origin: New origin after cropping.
            new_width: New width after cropping.
            new_height: New height after cropping.

        Returns:
            box_coordinates: New coordinates for the bounding box after cropping.
            inside_mask: Boolean mask to indicate whether the box is inside the image boundaries.
        """

        # Transform coordinates to cropped part of the image.
        box_coordinates[:, [0, 2]] -= x_crop_origin
        box_coordinates[:, [1, 3]] -= y_crop_origin

        # Scale coordinates to original image size.
        box_coordinates[:, [0, 2]] *= self.image_size[0] / new_width
        box_coordinates[:, [1, 3]] *= self.image_size[1] / new_height

        box_coordinates[:, [0, 2]] = torch.maximum(box_coordinates[:, [0, 2]], torch.tensor(0))
        box_coordinates[:, [0, 2]] = torch.minimum(box_coordinates[:, [0, 2]], torch.tensor(self.image_size[0]))
        box_coordinates[:, [1, 3]] = torch.maximum(box_coordinates[:, [1, 3]], torch.tensor(0))
        box_coordinates[:, [1, 3]] = torch.minimum(box_coordinates[:, [1, 3]], torch.tensor(self.image_size[1]))

        inside_mask = torch.greater_equal(_calculate_batched_box_area(box_coordinates), self.min_area)

        return box_coordinates, inside_mask


def _calculate_batched_box_area(box):
    return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
