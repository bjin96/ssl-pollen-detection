import torch
from torch import Tensor
from torch.nn.functional import one_hot, softmax


def calculate_focal_loss(
        class_logits: Tensor,
        labels: Tensor,
        alpha: float = 0.25,
        gamma: float = 2.,
        reduction: str = 'mean',
        weight: Tensor = None,
):
    """
    Implements the focal loss for classification as described in "Focal Loss for Dense Object Detection" by Lin et. al.,
    see https://arxiv.org/abs/1708.02002.

    Args:
        class_logits: Predicted class logits.
        labels: Ground-truth class labels.
        alpha: Balancing factor.
        gamma: Focusing parameter.
        reduction: Reduction operation.
        weight: Tensor with weights for each of the classes.

    Returns:
        Focal loss.
    """
    epsilon = 10e-5

    one_hot_labels = one_hot(labels, num_classes=16)
    class_probabilities = softmax(class_logits, dim=-1)
    p_t = torch.sum(class_probabilities * one_hot_labels, dim=-1) + epsilon
    focal_loss = -torch.log(p_t) * ((1 - p_t) ** gamma)

    if alpha >= 0:
        focal_loss = alpha * focal_loss

    if weight is not None:
        focal_loss = torch.sum(one_hot_labels * weight, dim=-1) * focal_loss

    if reduction == 'mean':
        focal_loss = focal_loss.mean()
    elif reduction == 'sum':
        focal_loss = focal_loss.sum()

    return focal_loss
