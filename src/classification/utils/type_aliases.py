from typing import TypeAlias, TypeVar
import numpy as np
import numpy.typing as npt
import torch

LabelDtype: TypeAlias = np.uint32
LabelArray: TypeAlias =  npt.NDArray[LabelDtype]
StandardizedImgDType: TypeAlias = np.float64
ImgDType: TypeAlias = StandardizedImgDType | np.uint8
ImgRaw: TypeAlias = npt.NDArray[np.uint8]
ImgStandardized: TypeAlias = npt.NDArray[np.float64]
ImgArray: TypeAlias = ImgRaw | ImgStandardized
ImgArrayT = TypeVar("ImgArrayT", ImgRaw, ImgStandardized)
ArrayOrTensor: TypeAlias = npt.NDArray[np.float64] | torch.Tensor
