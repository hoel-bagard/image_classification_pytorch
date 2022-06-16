import itertools
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.torch_utils.utils.batch_generator import BatchGenerator
from src.torch_utils.utils.metrics import Metrics
from src.torch_utils.utils.misc import clean_print


class ClassificationMetrics(Metrics):
    """Class computing usefull metrics for classification like tasks."""
    def __init__(self,
                 model: nn.Module,
                 train_dataloader: BatchGenerator,
                 val_dataloader: BatchGenerator,
                 label_map: dict[int, str],
                 max_batches: Optional[int] = 10):
        """Initialize the instance.

        Args:
            model (nn.Module): The PyTorch model being trained
            train_dataloader (BatchGenerator): DataLoader containing train data
            val_dataloader (BatchGenerator): DataLoader containing validation data
            label_map (dict): Dictionary linking class index to class name
            max_batches (int): If not None, then the metrics will be computed using at most this number of batches
        """
        super().__init__(model, train_dataloader, val_dataloader, max_batches)

        self.cm: np.ndarray  # Confusion Matrix
        self.label_map = label_map
        self.nb_output_classes = len(label_map)

    def compute_confusion_matrix(self, mode: str = "Train"):
        """Computes the confusion matrix. This function has to be called before using the get functions.

        Args:
            mode (str): Either "Train" or "Validation"
        """
        self.cm = np.zeros((self.nb_output_classes, self.nb_output_classes))
        dataloader = self.train_dataloader if mode == "Train" else self.val_dataloader
        for step, batch in enumerate(dataloader, start=1):
            data_batch, labels_batch = batch[0].float(), batch[1]
            predictions_batch = self.model(data_batch.to(self.device))

            predictions_batch = torch.argmax(predictions_batch, dim=-1).int().cpu().detach().numpy()
            for (label, pred) in zip(labels_batch, predictions_batch):
                self.cm[label, pred] += 1

            if self.max_batches and step >= self.max_batches:
                break
        dataloader.reset_epoch()  # Reset the epoch to not cause issues for other functions

    def get_avg_acc(self) -> float:
        """Uses the confusion matrix to return the average accuracy of the model.

        Returns:
            float: Average accuracy
        """
        avg_acc = np.sum([self.cm[i, i] for i in range(len(self.cm))]) / np.sum(self.cm)
        return avg_acc

    def get_class_accuracy(self) -> list[float]:
        """Uses the confusion matrix to return the average accuracy of the model.

        Returns:
            list: An array containing the accuracy for each class
        """
        per_class_acc = [self.cm[i, i] / max(1, np.sum(self.cm[i])) for i in range(len(self.cm))]
        return per_class_acc

    def get_class_iou(self) -> list[float]:
        """Uses the confusion matrix to return the iou for each class.

        Returns:
            list: List of the IOU for each class
        """
        intersections = [self.cm[i, i] for i in range(len(self.cm))]
        unions = [np.sum(self.cm[i, :]) + np.sum(self.cm[:, i]) - self.cm[i, i] for i in range(self.nb_output_classes)]
        per_class_iou = [intersections[i] / unions[i] for i in range(self.nb_output_classes)]
        return per_class_iou

    def get_confusion_matrix(self) -> np.ndarray:
        """Returns an image containing the plotted confusion matrix.

        Taken from: https://towardsdatascience.com/exploring-confusion-matrix-evolution-on-tensorboard-e66b39f4ac12

        Returns:
            np.ndarray: Image of the confusion matrix.
        """
        # Normalize the confusion matrix.
        cm = np.around(self.cm.astype("float") / self.cm.sum(axis=1)[:, np.newaxis], decimals=2)
        class_names = self.label_map.values()

        fig = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel("True label", labelpad=-5)
        plt.xlabel("Predicted label")
        fig.canvas.draw()

        # Convert matplotlib plot to normal image
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)  # Close figure explicitly to avoid memory leak

        return img

    def get_metrics(self, mode: str = "Train", **kwargs) -> dict[str, dict[str, Any]]:
        """See base class."""
        metrics: dict[str, dict] = {"scalars": {}, "imgs": {}}

        clean_print("Computing confusion matrix", end="\r")
        self.compute_confusion_matrix(mode=mode)

        clean_print("Computing average accuracy", end="\r")
        avg_acc = self.get_avg_acc()
        metrics["scalars"]["Average Accuracy"] = avg_acc

        clean_print("Computing per class accuracy", end="\r")
        per_class_acc = self.get_class_accuracy()
        for key, acc in enumerate(per_class_acc):
            metrics["scalars"][f"Per Class Accuracy/{self.label_map[key]}"] = acc

        clean_print("Creating confusion matrix image", end="\r")
        confusion_matrix = self.get_confusion_matrix()
        metrics["imgs"]["Confusion Matrix"] = confusion_matrix

        return metrics
