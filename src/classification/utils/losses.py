import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class SmoothCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing: float = 0) -> None:
        """Compute a loss with label smoothing for the given batch.

        Args:
            smoothing: Value used for label smoothing

        """
        super().__init__()
        if not 0 <= smoothing < 1:
            msg = "Smoothing value must be in [0, 1[."
            raise ValueError(msg)
        self.smoothing = smoothing

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, nb_classes: int, smoothing: float = 0.0) -> torch.Tensor:
        """Smooth the target labels.

        Args:
            targets: The target labels (ints, not one hots). Shape (batch_size, ).
            nb_classes: The number of output classes.
            smoothing: Value used for label smoothing.

        Returns:
            torch.Tensor: The smoothed labels with shape (batch_size, nb_classes).

        """
        with torch.no_grad():
            targets = (
                torch.empty(size=(targets.size(0), nb_classes), device=targets.device)
                .fill_(smoothing / (nb_classes - 1))
                .scatter_(1, targets.data.unsqueeze(1), 1.0 - smoothing)
            )
        return targets

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            logits: The model's predictions
            targets: The labels

        Returns:
            float: the loss

        """
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, logits.size(-1), self.smoothing)
        logits = F.log_softmax(logits, -1)
        loss = torch.mean(torch.sum(-targets * logits, dim=-1))
        return loss
