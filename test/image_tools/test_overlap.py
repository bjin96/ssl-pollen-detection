from unittest import TestCase
from unittest.mock import patch, Mock, call

import torch

from src.image_tools.overlap import batched_area_calculation, clean_pseudo_labels


class TestOverlap(TestCase):

    @patch(
        'src.image_tools.overlap.batched_area_calculation',
        side_effect=[
            torch.Tensor([100, 400]),
            torch.Tensor([100, 400])
        ]
    )
    def test_clean_pseudo_labels_with_partial_overlap(self, batched_area_calculation_mock: Mock):
        raw_x_pseudo = [
            {
                'boxes': torch.Tensor([[0.5, 0.5, 10.5, 10.5], [80, 80, 120, 120]]),
                'labels': torch.Tensor([1, 2]),
                'scores': torch.Tensor([0.7, 0.8])
            },
            {
                'boxes': torch.Tensor([[70, 70, 110, 110], [21, 21, 41, 41]]),
                'labels': torch.Tensor([3, 4]),
                'scores': torch.Tensor([0.6, 0.3])
            }
        ]
        y = [
            {
                'boxes': torch.Tensor([[0, 0, 10, 10], [100, 100, 140, 140]]),
                'labels': torch.Tensor([1, 2]),
                'image_id': torch.Tensor([123]),
                'area': torch.Tensor([100, 1600]),
                'iscrowd': torch.Tensor([0, 0]),
            },
            {
                'boxes': torch.Tensor([[100, 100, 140, 140], [20, 20, 40, 40]]),
                'labels': torch.Tensor([3, 4]),
                'image_id': torch.Tensor([124]),
                'area': torch.Tensor([1600, 400]),
                'iscrowd': torch.Tensor([0, 0])
            }
        ]
        expected_boxes = [
            torch.Tensor([[0, 0, 10, 10], [100, 100, 140, 140], [80, 80, 120, 120]]),
            torch.Tensor([[100, 100, 140, 140], [20, 20, 40, 40], [70, 70, 110, 110]])
        ]
        expected_labels = [
            torch.Tensor([1, 2, 2]),
            torch.Tensor([3, 4, 3]),
        ]
        expected_image_id = [
            torch.Tensor([123]),
            torch.Tensor([124]),
        ]
        expected_area = [
            torch.Tensor([100, 1600, 400]),
            torch.Tensor([1600, 400, 100]),
        ]
        expected_is_crowd = [
            torch.tensor([0, 0, 0], dtype=torch.uint8),
            torch.tensor([0, 0, 0], dtype=torch.uint8)
        ]
        expected_scores = [
            torch.Tensor([1., 1., 0.8]),
            torch.Tensor([1., 1., 0.6]),
        ]
        actual_cleaned_labels = clean_pseudo_labels(raw_x_pseudo, y)

        for index, actual_cleaned_label in enumerate(actual_cleaned_labels):
            torch.testing.assert_close(expected_boxes[index], actual_cleaned_label['boxes'])
            torch.testing.assert_close(expected_labels[index], actual_cleaned_label['labels'])
            torch.testing.assert_close(expected_image_id[index], actual_cleaned_label['image_id'])
            torch.testing.assert_close(expected_area[index], actual_cleaned_label['area'])
            torch.testing.assert_close(expected_is_crowd[index], actual_cleaned_label['iscrowd'])
            torch.testing.assert_close(expected_scores[index], actual_cleaned_label['scores'])

        batched_area_calculation_mock.assert_has_calls(
            [call(sample['boxes']) for sample in raw_x_pseudo]
        )

    def test_batched_area_calculation(self):
        boxes = torch.Tensor([[0, 0, 10, 10], [5, 5, 15, 15], [40, 40, 50, 50]])
        areas = torch.Tensor([100, 100, 100])

        torch.testing.assert_close(batched_area_calculation(boxes), areas)
