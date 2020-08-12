import cv2
import numpy as np


def draw_pred(imgs: np.ndarray, predictions: np.ndarray, labels: np.ndarray):
    """
    Draw predictions and labels on the image to help with TensorBoard visualisation.
    Args:
        imgs: Raw images.
        predictions: Predictions of the network, after softmax but before taking argmax
        labels: Labels corresponding to the images
    Returns: images with information written on them
    """
    new_imgs = []
    for img, preds, label in zip(imgs, predictions, labels):
        img = np.asarray(img * 255.0, dtype=np.uint8)
        img = cv2.resize(img, (480, 480), interpolation=cv2.INTER_AREA)
        preds = str([round(float(conf), 2) for conf in preds]) + f"  ==> {np.argmax(preds)}"
        img = cv2.putText(img, preds, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
        new_imgs.append(cv2.putText(img, f"Label: {label}", (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA))
    return np.asarray(new_imgs)
