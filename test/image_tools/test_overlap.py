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
            {'boxes': torch.Tensor([[0, 0, 10, 10], [20, 20, 40, 40]])},
            {'boxes': torch.Tensor([[0, 0, 10, 10], [20, 20, 40, 40]])}
        ]
        y = [
            {'boxes': torch.Tensor([[0, 0, 10, 10], [100, 100, 140, 140]])},
            {'boxes': torch.Tensor([[100, 100, 140, 140], [20, 20, 40, 40]])}
        ]
        expected_boxes = [
            torch.Tensor([[0, 0, 10, 10], [100, 100, 140, 140], [20, 20, 40, 40]]),
            torch.Tensor([[100, 100, 140, 140], [20, 20, 40, 40], [0, 0, 10, 10]])
        ]
        actual_cleaned_labels = clean_pseudo_labels(raw_x_pseudo, y)

        for expected_box, actual_cleaned_label in zip(expected_boxes, actual_cleaned_labels):
            torch.testing.assert_equal(expected_box, actual_cleaned_label['boxes'])

        calculate_box_overlapping_mock.assert_called()

    def test_calculate_box_overlapping(self):
        box1 = torch.Tensor([0, 0, 10, 10])
        box2 = torch.Tensor([5, 5, 15, 15])
        box3 = torch.Tensor([40, 40, 50, 50])

        self.assertTrue(calculate_box_overlapping(box1, box2))
        self.assertTrue(calculate_box_overlapping(box2, box1))
        self.assertFalse(calculate_box_overlapping(box1, box3))
        self.assertFalse(calculate_box_overlapping(box3, box1))
