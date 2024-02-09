from __future__ import annotations

from typing import Any, TYPE_CHECKING

import timm
import torch

from .cnn import CNN
from .small_darknet import SmallDarknet

if TYPE_CHECKING:
    from pathlib import Path


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
    model_path: Path | None = None,
    *,
    use_timm_pretrain: bool = True,
    eval_mode: bool = False,
    **kwargs: dict[str, Any],  # TODO: Have a typed dict
) -> torch.nn.Module:
    """Instantiate the given model.

    Args:
        model_name: Class of the model to instanciates
        nb_classes: Number of classes in the dataset
        model_path: If given, then the weights will be loaded from that checkpoint
        use_timm_pretrain: If using a timm model, whether to use a pretrain or not.
        eval_mode: Whether the model will be used for evaluation or not
        kwargs: Must contain image_sizes and nb_classes

    Returns:
        Instantiated PyTorch model
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
