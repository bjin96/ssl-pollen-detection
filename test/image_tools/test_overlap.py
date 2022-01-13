from unittest import TestCase
from unittest.mock import patch, Mock, call

import torch

from src.image_tools.overlap import calculate_box_overlapping, clean_pseudo_labels


class TestOverlap(TestCase):

    @patch(
        'src.image_tools.overlap.calculate_box_overlapping',
        side_effect=[
            True,
            False, False,
            False, False,
            False, True
        ]
    )
    def test_clean_pseudo_labels(self, calculate_box_overlapping_mock: Mock):
        raw_x_pseudo = [
            {
                'boxes': torch.Tensor([[0, 0, 10, 10], [20, 20, 40, 40]]),
                'labels': torch.Tensor([1, 2]),
                'scores': torch.Tensor([0.7, 0.8])
            },
            {
                'boxes': torch.Tensor([[0, 0, 10, 10], [20, 20, 40, 40]]),
                'labels': torch.Tensor([3, 4]),
                'scores': torch.Tensor([0.6, 0.3])
            }
        ]
        y = [
            {
                'boxes': torch.Tensor([[0, 0, 10, 10], [100, 100, 140, 140]]),
                'labels': torch.Tensor([5, 6]),
                'image_id': torch.Tensor([123]),
                'area': torch.Tensor([100, 1600]),
                'iscrowd': torch.Tensor([0, 0]),
            },
            {
                'boxes': torch.Tensor([[100, 100, 140, 140], [20, 20, 40, 40]]),
                'labels': torch.Tensor([7, 8]),
                'image_id': torch.Tensor([124]),
                'area': torch.Tensor([1600, 400]),
                'iscrowd': torch.Tensor([0, 0])
            }
        ]
        expected_boxes = [
            torch.Tensor([[0, 0, 10, 10], [100, 100, 140, 140], [20, 20, 40, 40]]),
            torch.Tensor([[100, 100, 140, 140], [20, 20, 40, 40], [0, 0, 10, 10]])
        ]
        expected_labels = [
            torch.Tensor([5, 6, 2]),
            torch.Tensor([7, 8, 3]),
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
            torch.Tensor([0, 0, 0]),
            torch.Tensor([0, 0, 0])
        ]
        expected_scores = [
            torch.Tensor([1., 1., 0.8]),
            torch.Tensor([1., 1., 0.6]),
        ]
        actual_cleaned_labels = clean_pseudo_labels(raw_x_pseudo, y)

        for index, actual_cleaned_label in enumerate(actual_cleaned_labels):
            torch.testing.assert_equal(expected_boxes[index], actual_cleaned_label['boxes'])
            torch.testing.assert_equal(expected_labels[index], actual_cleaned_label['labels'])
            torch.testing.assert_equal(expected_image_id[index], actual_cleaned_label['image_id'])
            torch.testing.assert_equal(expected_area[index], actual_cleaned_label['area'])
            torch.testing.assert_equal(expected_is_crowd[index], actual_cleaned_label['iscrowd'])
            torch.testing.assert_equal(expected_scores[index], actual_cleaned_label['scores'])

        calculate_box_overlapping_mock.assert_called()

    def test_calculate_box_overlapping(self):
        box1 = torch.Tensor([0, 0, 10, 10])
        box2 = torch.Tensor([5, 5, 15, 15])
        box3 = torch.Tensor([40, 40, 50, 50])

        self.assertTrue(calculate_box_overlapping(box1, box2))
        self.assertTrue(calculate_box_overlapping(box2, box1))
        self.assertFalse(calculate_box_overlapping(box1, box3))
        self.assertFalse(calculate_box_overlapping(box3, box1))
