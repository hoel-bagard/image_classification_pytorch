import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from config.data_config import DataConfig
from config.model_config import ModelConfig
from src.utils.draw import draw_pred
from src.utils.accuracy import (
    get_accuracy,
    get_class_accuracy
)


class TensorBoard():
    def __init__(self, model: nn.Module, max_outputs: int = 4):
        """
        Args:
            model: Model'whose performance are to be recorded
            max_outputs: Number of images kept and dislpayed in TensorBoard
        """
        super(TensorBoard, self).__init__()
        self.max_outputs = max_outputs
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.tb_writer = SummaryWriter(DataConfig.TB_DIR)
        self.tb_writer.add_graph(model, (torch.empty(1, 3, *ModelConfig.IMAGE_SIZES, device=self.device), ))
        self.tb_writer.flush()

    def write_images(self, epoch: int, dataloader: torch.utils.data.DataLoader, mode: str = "Train"):
        """
        Writes images with predictions written on them to TensorBoard
        Args:
            epoch: Current epoch
            dataloader: The images will be sampled from this dataset
            mode: Either "Train" or "Validation"
        """
        batch = next(iter(dataloader))
        in_imgs, labels = batch["img"][:self.max_outputs].float(), batch["label"][:self.max_outputs]
        predictions = self.model(in_imgs.to(self.device))
        predictions = torch.nn.functional.softmax(predictions, dim=-1)
        out_imgs = draw_pred(in_imgs, predictions, labels)
        for image_index, out_img in enumerate(out_imgs):
            out_img = np.transpose(out_img, (2, 0, 1))  # HWC -> CHW
            self.tb_writer.add_image(f"{mode}/prediction_{image_index}", out_img, global_step=epoch)

    def write_metrics(self, epoch: int, dataloader: torch.utils.data.DataLoader, mode: str = "Train") -> float:
        """
        Writes accuracy metrics in the TensorBoard
        Args:
            epoch: Current epoch
            dataloader: The accuracy will be calculated using this data.
            mode: Either "Train" or "Validation"
        Returns:
            avg_acc: Average accuracy
        """
        avg_acc = get_accuracy(self.model, dataloader)
        self.tb_writer.add_scalar(f"Average Accuracy/{mode}", avg_acc, epoch)
        per_class_acc = get_class_accuracy(self.model, dataloader)
        for key, acc in enumerate(per_class_acc):
            self.tb_writer.add_scalar(f"Per Class Accuracy/{mode} {DataConfig.LABEL_MAP[key]}", acc, epoch)

        return avg_acc

    def write_loss(self, epoch: int, loss: float, mode: str = "Train"):
        """
        Writes loss metric in the TensorBoard
        Args:
            epoch: Current epoch
            loss: Epoch loss that will be added to the TensorBoard
            mode: Either "Train" or "Validation"
        """
        self.tb_writer.add_scalar(f"Loss/{mode}", loss, epoch)
        self.tb_writer.flush()

    def write_lr(self, epoch: int, lr: float):
        """
        Writes learning rate in the TensorBoard
        Args:
            epoch: Current epoch
            lr: Learning rate for the given epoch
        """
        self.tb_writer.add_scalar("Learning Rate", lr, epoch)
