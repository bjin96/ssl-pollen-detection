import torch
from torchvision.ops import batched_nms


@torch.no_grad()
def clean_pseudo_labels(raw_x_pseudo, y, iou_threshold: float = 0.7):
    """
    Remove labels generated by the teacher that have any overlap with ground truth boxes.
    TODO: Use a defined IOU?
    """

    for predicted_sample, ground_truth_sample in zip(raw_x_pseudo, y):
        device = ground_truth_sample['boxes'].device
        boxes = torch.cat([ground_truth_sample['boxes'], predicted_sample['boxes']])
        scores = torch.cat([
            torch.ones(ground_truth_sample['boxes'].shape[0], device=device),
            predicted_sample['scores']
        ])
        labels = torch.cat([ground_truth_sample['labels'], predicted_sample['labels']])
        area = torch.cat([ground_truth_sample['area'], batched_area_calculation(predicted_sample['boxes'])])
        is_crowd = torch.zeros(len(boxes), dtype=torch.uint8, device=device)
        is_pseudo = torch.cat([
            torch.zeros(len(ground_truth_sample['boxes']), device=device),
            torch.ones(len(predicted_sample['boxes']), device=device),
        ])
        keep_indices = batched_nms(
            boxes=boxes,
            scores=scores,
            idxs=labels,
            iou_threshold=iou_threshold
        )
        ground_truth_sample['boxes'] = boxes[keep_indices]
        ground_truth_sample['scores'] = scores[keep_indices]
        ground_truth_sample['labels'] = labels[keep_indices]
        ground_truth_sample['area'] = area[keep_indices]
        ground_truth_sample['iscrowd'] = is_crowd[keep_indices]
        ground_truth_sample['is_pseudo'] = is_pseudo[keep_indices]
    return y


def batched_area_calculation(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
