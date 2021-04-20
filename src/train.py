import time
import os
from logging import Logger
from subprocess import CalledProcessError
import sys
import traceback

import torch
import torch.nn as nn

from config.data_config import DataConfig
from config.model_config import ModelConfig
from src.loss import SmoothCrossEntropyLoss
from src.torch_utils.utils.trainer import Trainer
from src.torch_utils.utils.tensorboard import TensorBoard
from src.torch_utils.utils.classification_metrics import ClassificationMetrics
from src.torch_utils.utils.batch_generator import BatchGenerator
from src.torch_utils.utils.ressource_usage import resource_usage


def train(model: nn.Module, train_dataloader: BatchGenerator, val_dataloader: BatchGenerator, logger: Logger):
    """ Trains and validate the given model using the datasets.

    Args:
        model (nn.Module): Model to train
        train_dataloader (BatchGenerator): BatchGenerator of the training data
        val_dataloader (BatchGenerator): BatchGenerator of the validation data
        logger: Logger used to print and / or save process outputs  to a log file
    """
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # weights = torch.Tensor(ModelConfig.LOSS_WEIGTHS).to(device) if ModelConfig.LOSS_WEIGTHS else None
    # loss_fn = nn.CrossEntropyLoss(weight=weights)
    loss_fn = SmoothCrossEntropyLoss(ModelConfig.LABEL_SMOOTHING)
    optimizer = torch.optim.Adam(model.parameters(), lr=ModelConfig.LR, weight_decay=ModelConfig.REG_FACTOR)
    trainer = Trainer(model, loss_fn, optimizer, train_dataloader, val_dataloader)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(trainer.optimizer, gamma=ModelConfig.LR_DECAY)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(trainer.optimizer, 10, T_mult=2)

    if DataConfig.USE_TB:
        metrics = ClassificationMetrics(model, train_dataloader, val_dataloader, DataConfig.LABEL_MAP, max_batches=None)
        tensorboard = TensorBoard(model, DataConfig.TB_DIR, ModelConfig.IMAGE_SIZES, metrics, DataConfig.LABEL_MAP,
                                  ModelConfig.GRAYSCALE)

    best_loss = 1000
    last_checkpoint_epoch = 0
    train_start_time = time.time()

    try:
        for epoch in range(ModelConfig.MAX_EPOCHS):
            epoch_start_time = time.perf_counter()
            print()  # logger doesn't handle \n super well
            logger.info(f"Epoch {epoch}/{ModelConfig.MAX_EPOCHS}")

            epoch_loss = trainer.train_epoch()

            if DataConfig.USE_TB:
                tensorboard.write_loss(epoch, epoch_loss)
                tensorboard.write_lr(epoch, scheduler.get_last_lr()[0])

            if (epoch_loss < best_loss and DataConfig.USE_CHECKPOINT and epoch >= DataConfig.RECORD_START
                    and (epoch - last_checkpoint_epoch) >= DataConfig.CHECKPT_SAVE_FREQ):
                save_path = os.path.join(DataConfig.CHECKPOINT_DIR, f"train_{epoch}.pt")
                logger.info(f"Loss improved from {best_loss:.5e} to {epoch_loss:.5e},"
                            f"saving model to {save_path}")
                best_loss, last_checkpoint_epoch = epoch_loss, epoch
                torch.save(model.state_dict(), save_path)

            logger.info(f"Epoch loss: {epoch_loss:.5e}  -  Took {time.perf_counter() - epoch_start_time:.5f}s")

            # Validation and other metrics
            if epoch % DataConfig.VAL_FREQ == 0 and epoch >= DataConfig.RECORD_START:
                with torch.no_grad():
                    validation_start_time = time.perf_counter()
                    epoch_loss = trainer.val_epoch()

                    if DataConfig.USE_TB:
                        print("Starting to compute TensorBoard metrics", end="\r", flush=True)
                        # TODO: uncomment this after finishing the lambda network
                        # tensorboard.write_weights_grad(epoch)
                        tensorboard.write_loss(epoch, epoch_loss, mode="Validation")

                        # Metrics for the Train dataset
                        tensorboard.write_images(epoch, train_dataloader)
                        tensorboard.write_metrics(epoch)
                        train_acc = metrics.get_avg_acc()

                        # Metrics for the Validation dataset
                        tensorboard.write_images(epoch, val_dataloader, mode="Validation")
                        tensorboard.write_metrics(epoch, mode="Validation")
                        val_acc = metrics.get_avg_acc()

                        logger.info(f"Train accuracy: {train_acc:.3f}  -  Validation accuracy: {val_acc:.3f}")

                    logger.info(f"Validation loss: {epoch_loss:.5e}  -  "
                                f"Took {time.perf_counter() - validation_start_time:.5f}s")
            scheduler.step()
    except KeyboardInterrupt:
        print("\n")
    except Exception as error:
        logger.error(''.join(traceback.format_exception(*sys.exc_info())))
        raise error

    if DataConfig.USE_TB:
        tensorboard.close_writers()

    train_stop_time = time.time()
    end_msg = f"Finished Training\n\tTraining time : {train_stop_time - train_start_time:.03f}s"
    try:
        memory_peak, gpu_memory = resource_usage()
        end_msg += f"\n\tRAM peak : {memory_peak // 1024} MB\n\tVRAM usage : {gpu_memory}"
    except CalledProcessError:
        pass
    logger.info(end_msg)
