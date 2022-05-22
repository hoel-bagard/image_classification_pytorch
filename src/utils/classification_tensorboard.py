from logging import Logger
from pathlib import Path
from typing import Callable

import numpy as np
import numpy.typing as npt
import torch.nn as nn

from config.data_config import get_data_config
from config.model_config import get_model_config
from src.torch_utils.utils.batch_generator import BatchGenerator
from src.torch_utils.utils.misc import clean_print
from src.torch_utils.utils.tensorboard_template import TensorBoard
from src.utils.classification_metrics import ClassificationMetrics
from src.utils.draw_functions import draw_pred_img


class ClassificationTensorBoard(TensorBoard):
    """Class with TensorBoard functions for classification.

    Args:
        model (nn.Module): Pytorch model whose performance are to be recorded
        tb_dir (Path): Path to where the tensorboard files will be saved
        train_dataloader (BatchGenerator): DataLoader with a PyTorch DataLoader like interface, contains train data
        val_dataloader (BatchGenerator): DataLoader containing  validation data
        logger (Logger): Used to print things.
        metrics (Metrics, optional): Instance of the Metrics class, used to compute classification metrics
        denormalize_imgs_fn (Callable): Function to destandardize a batch of images.
        write_graph (bool): If True, add the network graph to the TensorBoard
        max_outputs (int): Maximal number of images kept and displayed in TensorBoard (per function call)
    """
    def __init__(self,
                 model: nn.Module,
                 tb_dir: Path,
                 train_dataloader: BatchGenerator,
                 val_dataloader: BatchGenerator,
                 logger: Logger,
                 metrics: ClassificationMetrics,
                 denormalize_imgs_fn: Callable,
                 write_graph: bool = True,
                 max_outputs: int = 4):
        super().__init__(model, tb_dir, train_dataloader, val_dataloader, logger, metrics, write_graph)
        self.max_outputs = max_outputs
        self.denormalize_imgs_fn = denormalize_imgs_fn
        self.model_config = get_model_config()
        self.data_config = get_data_config()

    def write_images(self, epoch: int, mode: str = "Train"):
        """Writes images with predictions written on them to TensorBoard.

        Args:
            epoch (int): Current epoch
            mode (str): Either "Train" or "Validation"
        """
        clean_print("Writing images", end="\r")
        tb_writer = self.train_tb_writer if mode == "Train" else self.val_tb_writer
        dataloader = self.train_dataloader if mode == "Train" else self.val_dataloader

        batch = dataloader.next_batch()
        imgs, labels = batch[0][:self.max_outputs], batch[1][:self.max_outputs]

        # Get some predictions
        predictions = self.model(imgs)

        imgs: npt.NDArray[np.uint8] = self.denormalize_imgs_fn(imgs)
        labels = labels.cpu().detach().numpy()
        predictions = predictions.cpu().detach().numpy()
        out_imgs = draw_pred_img(imgs, predictions, labels, self.data_config.LABEL_MAP)

        # Add them to TensorBoard
        for image_index, out_img in enumerate(out_imgs):
            tb_writer.add_image(f"{mode}/prediction_{image_index}", out_img, global_step=epoch, dataformats="HWC")

        dataloader.reset_epoch()  # Reset the epoch to not cause issues for other functions

    def write_losses(self, epoch: int, losses: list[float], names: list[str], mode: str = "Train"):
        """Writes loss metric in TensorBoard.

        Args:
            epoch (int): Current epoch
            losses: Losses to add to the TensorBoard
            names: Name for each loss
            mode (str): Either "Train" or "Validation"
        """
        tb_writer = self.train_tb_writer if mode == "Train" else self.val_tb_writer
        for name, loss in zip(names, losses):
            tb_writer.add_scalar(f"Loss Components/{name}", loss, epoch)
        self.train_tb_writer.flush()
