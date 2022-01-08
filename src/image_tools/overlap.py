import torch


@torch.no_grad()
def clean_pseudo_labels(raw_x_pseudo, y):
    """
    Remove labels generated by the teacher that have any overlap with ground truth boxes.
    """
    for predicted_sample, ground_truth_sample in zip(raw_x_pseudo, y):
        non_overlapping_pseudo_boxes = []
        for pseudo_box in predicted_sample['boxes']:
            is_overlapping = False
            for ground_truth_box in ground_truth_sample['boxes']:
                if calculate_box_overlapping(pseudo_box, ground_truth_box):
                    is_overlapping = True
                    break
            if not is_overlapping:
                non_overlapping_pseudo_boxes.append(pseudo_box)
        non_overlapping_pseudo_boxes = torch.stack(non_overlapping_pseudo_boxes).to(ground_truth_sample['boxes'].device)
        ground_truth_sample['boxes'] = torch.cat([ground_truth_sample['boxes'], non_overlapping_pseudo_boxes])
    return y


def calculate_box_overlapping(box_1, box_2):
    return box_1[2] >= box_2[0] and box_2[2] >= box_1[0] and box_1[3] >= box_2[1] and box_2[3] >= box_1[1]