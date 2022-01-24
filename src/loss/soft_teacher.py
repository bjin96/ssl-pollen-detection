import torch
from torch.nn.functional import softmax, cross_entropy


def soft_teacher_classification_loss(
        class_logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_background_scores: torch.Tensor,
        is_pseudo: torch.Tensor,
        unsupervised_loss_weight: float = 1.0
) -> torch.Tensor:
    epsilon = 10e-5
    class_probabilities = softmax(class_logits, -1)
    example_argmax = torch.argmax(class_probabilities, -1)
    select_foreground = example_argmax > 0

    unweighted_loss = cross_entropy(class_logits, labels, reduction='none')

    select_supervised_background = torch.logical_and(torch.logical_not(select_foreground), torch.logical_not(is_pseudo))
    select_unsupervised_background = torch.logical_and(torch.logical_not(select_foreground), is_pseudo)
    reliability_weight = teacher_background_scores[select_unsupervised_background] \
                         / (torch.sum(teacher_background_scores[select_unsupervised_background]) + epsilon)
    supervised_background_loss = torch.mean(unweighted_loss[select_supervised_background])
    unsupervised_background_loss = torch.mean(reliability_weight * unweighted_loss[select_unsupervised_background])

    select_supervised_foreground = torch.logical_and(select_foreground, torch.logical_not(is_pseudo))
    select_unsupervised_foreground = torch.logical_and(select_foreground, is_pseudo)
    supervised_foreground_loss = torch.mean(unweighted_loss[select_supervised_foreground])
    unsupervised_foreground_loss = torch.mean(unweighted_loss[select_unsupervised_foreground])

    # TODO: Maybe divide by 4 to not destroy equilibrium between classification and regression loss (own addition)
    supervised_loss = supervised_foreground_loss + supervised_background_loss
    unsupervised_loss = unsupervised_foreground_loss + unsupervised_background_loss
    # TODO: Maybe additional hyperparameter to weight unsupervised loss?
    return supervised_loss + unsupervised_loss_weight * unsupervised_loss
