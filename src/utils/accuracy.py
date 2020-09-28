import numpy as np
import torch
import torch.nn as nn

from config.model_config import ModelConfig


def get_accuracy(model: nn.Module, dataloader: torch.utils.data.DataLoader, max_batches: int = 10):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    acc = 0
    for step, batch in enumerate(dataloader, start=1):
        imgs, labels = batch["img"].float(), batch["label"]
        predictions = model(imgs.to(device))
        predictions = torch.nn.functional.softmax(predictions, dim=-1)
        acc += torch.mean(torch.eq(labels.to(device), torch.argmax(predictions, dim=-1)).float())
        if step >= max_batches:
            break
    return acc / step


def get_class_accuracy(model: nn.Module, dataloader: torch.utils.data.DataLoader, max_batches: int = 10):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    acc = np.zeros(ModelConfig.OUTPUT_CLASSES)
    class_count = np.zeros(ModelConfig.OUTPUT_CLASSES)
    for step, batch in enumerate(dataloader, start=1):
        imgs, labels = batch["img"].float(), batch["label"]
        predictions = model(imgs.to(device))
        predictions = torch.nn.functional.softmax(predictions, dim=-1)

        correct_pred = torch.eq(labels.to(device), torch.argmax(predictions, dim=-1)).float().cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

        for (label, p) in zip(labels, correct_pred):
            acc[label] += p
            class_count[label] += 1

        if step >= max_batches:
            break
    return acc / class_count
