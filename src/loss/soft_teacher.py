from typing import Tuple, Callable

import torch


def soft_teacher_classification_loss(
        class_logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_background_scores: torch.Tensor,
        is_pseudo: torch.Tensor,
        classification_loss_function: Callable,
) -> Tuple[torch.Tensor, torch.Tensor]:
    select_foreground = labels > 0

    unweighted_loss = classification_loss_function(class_logits, labels, reduction='none')
    device = unweighted_loss.device

    supervised_loss = _calculate_supervised_loss(
        select_foreground=select_foreground,
        unweighted_loss=unweighted_loss,
        is_pseudo=is_pseudo,
        device=device,
    )
    unsupervised_loss = _calculate_unsupervised_loss(
        select_foreground=select_foreground,
        is_pseudo=is_pseudo,
        teacher_background_scores=teacher_background_scores,
        unweighted_loss=unweighted_loss,
        device=device,
    )

    return supervised_loss, unsupervised_loss


def _calculate_supervised_loss(
        select_foreground: torch.Tensor,
        unweighted_loss: torch.Tensor,
        is_pseudo: torch.Tensor,
        device,
):
    select_supervised_background = torch.logical_and(torch.logical_not(select_foreground), torch.logical_not(is_pseudo))
    if torch.numel(unweighted_loss[select_supervised_background]) > 0:
        supervised_background_loss = torch.mean(unweighted_loss[select_supervised_background])
    else:
        supervised_background_loss = torch.tensor(0., device=device)

    select_supervised_foreground = torch.logical_and(select_foreground, torch.logical_not(is_pseudo))
    if torch.numel(unweighted_loss[select_supervised_foreground]) > 0:
        supervised_foreground_loss = torch.mean(unweighted_loss[select_supervised_foreground])
    else:
        supervised_foreground_loss = torch.tensor(0., device=device)

    return supervised_background_loss + supervised_foreground_loss


def _calculate_unsupervised_loss(
        select_foreground: torch.Tensor,
        is_pseudo: torch.Tensor,
        teacher_background_scores: torch.Tensor,
        unweighted_loss: torch.Tensor,
        device,
):
    unsupervised_background_loss = _calculate_unsupervised_background_loss(
        select_foreground=select_foreground,
        is_pseudo=is_pseudo,
        teacher_background_scores=teacher_background_scores,
        unweighted_loss=unweighted_loss,
        device=device,
    )
    unsupervised_foreground_loss = _calculate_unsupervised_foreground_loss(
        select_foreground=select_foreground,
        is_pseudo=is_pseudo,
        unweighted_loss=unweighted_loss,
        device=device,
    )

    return unsupervised_foreground_loss + unsupervised_background_loss


def _calculate_unsupervised_background_loss(
        select_foreground: torch.Tensor,
        is_pseudo: torch.Tensor,
        teacher_background_scores: torch.Tensor,
        unweighted_loss: torch.Tensor,
        device,
):
    epsilon = 10e-5

    select_unsupervised_background = torch.logical_and(torch.logical_not(select_foreground), is_pseudo)

    reliability_weight = teacher_background_scores[select_unsupervised_background] \
                         / (torch.sum(teacher_background_scores[select_unsupervised_background]) + epsilon)

    if torch.numel(unweighted_loss[select_unsupervised_background]) > 0:
        unsupervised_background_loss = torch.sum(reliability_weight * unweighted_loss[select_unsupervised_background])
    else:
        unsupervised_background_loss = torch.tensor(0., device=device)

    return unsupervised_background_loss


def _calculate_unsupervised_foreground_loss(
        select_foreground: torch.Tensor,
        is_pseudo: torch.Tensor,
        unweighted_loss: torch.Tensor,
        device,
):
    select_unsupervised_foreground = torch.logical_and(select_foreground, is_pseudo)
    if torch.numel(unweighted_loss[select_unsupervised_foreground]) > 0:
        unsupervised_foreground_loss = torch.mean(unweighted_loss[select_unsupervised_foreground])
    else:
        unsupervised_foreground_loss = torch.tensor(0., device=device)

    return unsupervised_foreground_loss
