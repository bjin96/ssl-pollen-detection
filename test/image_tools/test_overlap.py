from unittest import TestCase
from unittest.mock import patch, Mock, call

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
            {'boxes': [[0, 0, 10, 10], [20, 20, 40, 40]]},
            {'boxes': [[0, 0, 10, 10], [20, 20, 40, 40]]}
        ]
        y = [
            {'boxes': [[0, 0, 10, 10], [100, 100, 140, 140]]},
            {'boxes': [[100, 100, 140, 140], [20, 20, 40, 40]]}
        ]
        expected_cleaned_labels = [
            {'boxes': [[0, 0, 10, 10], [100, 100, 140, 140], [20, 20, 40, 40]]},
            {'boxes': [[100, 100, 140, 140], [20, 20, 40, 40], [0, 0, 10, 10]]}
        ]
        actual_cleaned_labels = clean_pseudo_labels(raw_x_pseudo, y)
        self.assertListEqual(expected_cleaned_labels, actual_cleaned_labels)

        calculate_box_overlapping_mock.assert_has_calls(
            [
                call([0, 0, 10, 10], [0, 0, 10, 10]),
                call([20, 20, 40, 40], [0, 0, 10, 10]), call([20, 20, 40, 40], [100, 100, 140, 140]),
                call([0, 0, 10, 10], [100, 100, 140, 140]), call([0, 0, 10, 10], [20, 20, 40, 40]),
                call([20, 20, 40, 40], [100, 100, 140, 140]), call([20, 20, 40, 40], [20, 20, 40, 40])
            ]
        )

    def test_calculate_box_overlapping(self):
        box1 = [0, 0, 10, 10]
        box2 = [5, 5, 15, 15]
        box3 = [40, 40, 50, 50]

        self.assertTrue(calculate_box_overlapping(box1, box2))
        self.assertTrue(calculate_box_overlapping(box2, box1))
        self.assertFalse(calculate_box_overlapping(box1, box3))
        self.assertFalse(calculate_box_overlapping(box3, box1))
