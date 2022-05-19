from typing import Optional

import cv2
import numpy as np
import torch
from einops import rearrange


def draw_pred_img(imgs: torch.Tensor,
                  predictions: torch.Tensor,
                  labels: torch.Tensor,
                  label_map: dict[int, str],
                  size: Optional[tuple[int, int]] = None) -> np.ndarray:
    """Draws predictions and labels on the image to help with TensorBoard visualisation.

    Args:
        imgs (torch.Tensor): Raw imgs.
        predictions (torch.Tensor): Predictions of the network, after softmax but before taking argmax
        labels (torch.Tensor): Labels corresponding to the images
        label_map (dict): Dictionary linking class index to class name
        size (tuple, optional): If given, the images will be resized to this size

    Returns:
        np.ndarray: images with information written on them
    """
    imgs: np.ndarray = imgs.cpu().detach().numpy()
    labels: np.ndarray = labels.cpu().detach().numpy()
    predictions: np.ndarray = predictions.cpu().detach().numpy()

    imgs = rearrange(imgs, 'b c w h -> b w h c')  # imgs.transpose(0, 2, 3, 1)

    out_imgs = []
    for img, preds, label in zip(imgs, predictions, labels):
        nb_to_keep = 3 if len(preds) > 3 else 2  # have at most 3 classes printed
        idx = np.argpartition(preds, -nb_to_keep)[-nb_to_keep:]  # Gets indices of top predictions
        idx = idx[np.argsort(preds[idx])][::-1]
        preds = str([label_map[i] + f":  {round(float(preds[i]), 2)}" for i in idx])

        img = np.asarray(img * 255.0, dtype=np.uint8)
        if size:
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        img = cv2.UMat(img)
        img = cv2.copyMakeBorder(img, 40, 0, 0, 0, cv2.BORDER_CONSTANT, None, 0)
        img = cv2.putText(img, preds, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        img = cv2.putText(img, f"Label: {label}  ({label_map[label]})", (20, 40),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        out_img = img.get()
        # If opencv resizes a grayscale image, it removes the channel dimension
        if out_img.ndim == 2:
            out_img = np.expand_dims(out_img, -1)
        out_imgs.append(out_img)
    return np.asarray(out_imgs)
