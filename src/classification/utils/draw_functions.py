from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt


def draw_pred_img(
    imgs: npt.NDArray[np.uint8],
    predictions: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8],
    label_map: dict[int, str],
    size: Optional[tuple[int, int]] = None,
) -> npt.NDArray[np.uint8]:
    """Draws predictions and labels on the image to help with TensorBoard visualisation.

    Args:
    ----
        imgs: Raw imgs.
        predictions: Predictions of the network, after softmax but before taking argmax
        labels: Labels corresponding to the images
        label_map: Dictionary linking class index to class name
        size: If given, the images will be resized to this size

    Returns:
    -------
        np.ndarray: images with information written on them

    """
    out_imgs = []
    for img, preds, label in zip(imgs, predictions, labels):
        nb_to_keep = 3 if len(preds) > 3 else 2  # have at most 3 classes printed
        idx = np.argpartition(preds, -nb_to_keep)[-nb_to_keep:]  # Gets indices of top predictions
        idx = idx[np.argsort(preds[idx])][::-1]
        preds = str([label_map[i] + f":  {round(float(preds[i]), 2)}" for i in idx])

        if size:
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        img = cv2.UMat(img)
        img = cv2.copyMakeBorder(img, 60, 0, 0, 0, cv2.BORDER_CONSTANT, None, 0)
        img = cv2.putText(img, preds, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        img = cv2.putText(
            img,
            f"Label: {label}  ({label_map[label]})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        out_img = img.get()
        # If opencv resizes a grayscale image, it removes the channel dimension
        if out_img.ndim == 2:
            out_img = np.expand_dims(out_img, -1)
        out_imgs.append(out_img)
    return np.asarray(out_imgs)
