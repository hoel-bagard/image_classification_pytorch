import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothCrossEntropyLoss(nn.Module):
    """ Computes a loss with label smoothing for the given batch. """
    def __init__(self, smoothing: float = 0):
        """
        Args:
            smoothing (float): Value used for label smoothing
        """
        super().__init__()
        if not 0 <= smoothing < 1:
            raise ValueError("Smoothing value must be in [0, 1[.")
        self.smoothing = smoothing

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, nb_classes: int, smoothing: float = 0.0) -> torch.Tensor:
        """ Smoothes the target labels.

        Args:
            targets (torch.Tensor): The target labels (ints, not one hots). Shape (batch_size, ).
            nb_classes (int): The number of output classes.
            smoothing (float): Value used for label smoothing.

        Returns:
            torch.Tensor: The smoothed labels with shape (batch_size, nb_classes).

        Examples:
            # TODO
        """
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), nb_classes), device=targets.device) \
                .fill_(smoothing / (nb_classes-1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1.0 - smoothing)
        return targets

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): The model's predictions
            labels (torch.Tensor): The labels

        Returns:
            float: the loss
        """
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, logits.size(-1), self.smoothing)
        logits = F.log_softmax(logits, -1)
        loss = torch.mean(torch.sum(-targets * logits, dim=-1))
        return loss
