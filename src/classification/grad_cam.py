from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch

import classification.data.data_transformations as transforms
from classification.configs import TrainConfig
from classification.data.dataset_loaders import dog_vs_cat_loader as data_loader
from classification.data.defeault_loader import default_load_data
from classification.networks.build_network import build_model
from classification.torch_utils.utils.draw import draw_pred_img
from classification.torch_utils.utils.misc import clean_print, get_config_as_dict


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("model_path", type=Path, help="Path to the checkpoint to use")
    parser.add_argument("data_path", type=Path, help="Path to the test dataset")
    args = parser.parse_args()

    classes_names_path: Path | None = args.classes_names_path
    classes_names: list[str] | None = args.classes_names

    if classes_names_path is not None:
        train_config = TrainConfig.from_classes_path(classes_names_path)
    elif classes_names is not None:
        train_config = TrainConfig.from_classes_names(classes_names)
    else:
        msg = "Either --classes_names_path or --classes_names must be provided"
        raise ValueError(msg)

    # Creates and load the model
    model = build_model(
        train_config.MODEL,
        train_config.NB_CLASSES,
        model_path=args.model_path,
        eval=True,
        **get_config_as_dict(TrainConfig),
    )
    print("Weights loaded", flush=True)

    data, labels, paths = data_loader(
        args.data_path, train_config.LABEL_MAP, data_preprocessing_fn=default_load_data, return_img_paths=True
    )
    base_cpu_pipeline = (transforms.resize(train_config.IMAGE_SIZES),)
    base_gpu_pipeline = (transforms.to_tensor(), transforms.normalize(labels_too=True))
    data_transformations = transforms.compose_transformations((*base_cpu_pipeline, *base_gpu_pipeline))
    print("\nData loaded", flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for img, label, img_path in zip(data, labels, paths):
        clean_print(f"Processing image {img_path}", end="\r")
        img, label = data_transformations([img], [label])

        # Feed the image to the model
        output = model(img)
        output = torch.nn.functional.softmax(output, dim=-1)

        # Get top prediction and turn it into a one hot
        prediction = output.argmax(dim=1)
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][prediction] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(device) * output)

        # Get gradients and activations
        model.zero_grad()
        one_hot.backward(retain_graph=True)
        grads_val = model.get_gradients()[-1].cpu().data.numpy()

        activations = model.get_activations()
        activations = activations.cpu().data.numpy()[0, :]

        # Make gradcam mask
        weights = np.mean(grads_val, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, train_config.IMAGE_SIZES)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        # Draw prediction (logits) on the image
        img = draw_pred_img(img, output, label, train_config.LABEL_MAP, size=train_config.IMAGE_SIZES)

        # Fuse input image and gradcam mask
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap)
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)

        while True:
            cv2.imshow("Image", cam)
            key = cv2.waitKey(10)
            if key == ord("q"):
                cv2.destroyAllWindows()
                break


if __name__ == "__main__":
    main()
