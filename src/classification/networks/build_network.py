from __future__ import annotations

from enum import Enum
from typing import Any, TYPE_CHECKING

import timm
import torch

from classification.networks.cnn import CNN
from classification.networks.small_darknet import SmallDarknet

if TYPE_CHECKING:
    from pathlib import Path


class ModelHelper(str, Enum):
    """Helper enum given a list of available models (more timm models can be added).

    TODO: 3.11 | Use StrEnum if python version requirements becomes >=3.11
    """

    SmallDarknet = "small_darknet"
    CustomCNN = "custom_cnn"
    # Timm models below
    ConvNeXt = "convnext_small"
    ConvNeXtV2 = "convnextv2_small"
    ResNet34 = "resnet34"
    ResNet50 = "resnet50"
    ResNetv2_50t = "resnetv2_50t"
    EfficientVIT_M3 = "efficientvit_m3"
    MobileNetv3_small_050 = "mobilenetv3_small_050"
    EfficientNetv2_s = "efficientnetv2_s"

    def __str__(self) -> str:
        return self.value


def build_model(
    model_name: ModelHelper,
    nb_classes: int,
    model_path: Path | None = None,
    *,
    use_timm_pretrain: bool = True,
    eval_mode: bool = False,
    # TODO:  Really do it, spent 2 hours on it because eval vs eval_mode wasn't caught...
    **kwargs: Any,  # TODO: Have a typed dict.
) -> torch.nn.Module:
    """Instantiate the given model.

    Args:
        model_name: Class of the model to instantiates
        nb_classes: Number of classes in the dataset
        model_path: If given, then the weights will be loaded from that checkpoint
        use_timm_pretrain: If using a timm model, whether to use a pretrain or not.
        eval_mode: Whether the model will be used for evaluation or not
        kwargs: Must contain image_sizes and nb_classes

    Returns:
        Instantiated PyTorch model
    """
    # TODO: Take as an argument, do not hardcode.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    match model_name:
        case ModelHelper.SmallDarknet:
            kwargs["nb_classes"] = nb_classes
            model = SmallDarknet(**kwargs)
        case ModelHelper.CustomCNN:
            kwargs["nb_classes"] = nb_classes
            model = CNN(**kwargs)
        case _:
            model: torch.nn.Module = timm.create_model(model_name, num_classes=nb_classes, pretrained=use_timm_pretrain)

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device))
    if eval_mode:
        model.eval()

    model.to(device).float()
    return model
