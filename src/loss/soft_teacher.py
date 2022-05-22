from typing import Tuple, Callable

import torch
from torch.nn.functional import softmax


def soft_teacher_classification_loss(
        class_logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_background_scores: torch.Tensor,
        is_pseudo: torch.Tensor,
        classification_loss_function: Callable,
) -> Tuple[torch.Tensor, torch.Tensor]:
    epsilon = 10e-5
    class_probabilities = softmax(class_logits, -1)
    indices = torch.argmax(class_probabilities, -1)
    select_foreground = indices > 0

    unweighted_loss = classification_loss_function(class_logits, labels, reduction='none')

    select_supervised_background = torch.logical_and(torch.logical_not(select_foreground), torch.logical_not(is_pseudo))
    select_unsupervised_background = torch.logical_and(torch.logical_not(select_foreground), is_pseudo)
    reliability_weight = teacher_background_scores[select_unsupervised_background] \
                         / (torch.sum(teacher_background_scores[select_unsupervised_background]) + epsilon)
    device = unweighted_loss.device
    if torch.numel(unweighted_loss[select_supervised_background]) > 0:
        supervised_background_loss = torch.mean(unweighted_loss[select_supervised_background])
    else:
        supervised_background_loss = torch.tensor(0., device=device)
    if torch.numel(unweighted_loss[select_unsupervised_background]) > 0:
        unsupervised_background_loss = torch.mean(reliability_weight * unweighted_loss[select_unsupervised_background])
    else:
        unsupervised_background_loss = torch.tensor(0., device=device)

    select_supervised_foreground = torch.logical_and(select_foreground, torch.logical_not(is_pseudo))
    select_unsupervised_foreground = torch.logical_and(select_foreground, is_pseudo)
    if torch.numel(unweighted_loss[select_supervised_foreground]) > 0:
        supervised_foreground_loss = torch.mean(unweighted_loss[select_supervised_foreground])
    else:
        supervised_foreground_loss = torch.tensor(0., device=device)
    if torch.numel(unweighted_loss[select_unsupervised_foreground]) > 0:
        unsupervised_foreground_loss = torch.mean(unweighted_loss[select_unsupervised_foreground])
    else:
        unsupervised_foreground_loss = torch.tensor(0., device=device)

    # TODO: Maybe divide by 4 to not destroy equilibrium between classification and regression loss (own addition)
    supervised_loss = supervised_foreground_loss + supervised_background_loss
    unsupervised_loss = unsupervised_foreground_loss + unsupervised_background_loss

    return supervised_loss, unsupervised_loss
