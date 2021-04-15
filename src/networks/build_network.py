from typing import Optional
from pathlib import Path

import torch

from .small_darknet import SmallDarknet
from .lambda_network import LambdaResnet
from .cnn import CNN


class ModelHelper:
    SmallDarknet = SmallDarknet
    LambdaResnet = LambdaResnet
    CNN = CNN


def build_model(model_type: type, nb_classes: int, model_path: Optional[Path] = None,
                eval_mode: bool = False, **kwargs):
    """ Function that instanciates the given model.

    Args:
        model_type (type): Class of the model to instanciates
        nb_classes (int): Number of classes in the dataset
        model_path (Path): If given, then the weights will be loaded from that checkpoint
        eval (bool): Whether the model will be used for evaluation or not
    Returns:
        torch.nn.Module: Instantiated PyTorch model
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    kwargs["nb_classes"] = nb_classes
    model = model_type(**kwargs)

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device))
    if eval:
        model.eval()

    model.to(device).float()
    return model
