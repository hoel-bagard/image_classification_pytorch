from pathlib import Path
from typing import Any, Optional

import timm
import torch

from .cnn import CNN
from .small_darknet import SmallDarknet


class ModelHelper:
    SmallDarknet = SmallDarknet
    CNN = CNN
    ConvNeXt = "convnext_small"
    ResNetv2_50t = "resnetv2_50t"
    MobileNetv3_small_050 = "mobilenetv3_small_050"
    EfficientNetv2_s = "efficientnetv2_s"


def build_model(
    model_name: type | str,
    nb_classes: int,
    model_path: Optional[Path] = None,
    use_timm_pretrain: bool = True,
    eval_mode: bool = False,
    **kwargs: dict[str, Any],
) -> torch.nn.Module:
    """Function that instanciates the given model.

    Args:
    ----
        model_type (type): Class of the model to instanciates
        nb_classes (int): Number of classes in the dataset
        model_path (Path): If given, then the weights will be loaded from that checkpoint
        use_timm_pretrain (Bool): If using a timm model, whether to use a pretrain or not.
        eval (bool): Whether the model will be used for evaluation or not

    Returns:
    -------
        torch.nn.Module: Instantiated PyTorch model

    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if isinstance(model_name, str):
        model: torch.nn.Module = timm.create_model(model_name, num_classes=nb_classes, pretrained=use_timm_pretrain)
    else:
        kwargs["nb_classes"] = nb_classes
        model = model_name(**kwargs)

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device))
    if eval_mode:
        model.eval()

    model.to(device).float()
    return model
