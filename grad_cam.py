# import argparse
# import glob
# import os

# import cv2
# import torch
# from torchvision.transforms import Compose
# import numpy as np

# from config.data_config import DataConfig
# from config.model_config import ModelConfig
# from src.networks.build_network import build_model
# from src.utils.draw import draw_pred
# import src.dataset.transforms as transforms

from argparse import ArgumentParser
from pathlib import Path
import time

import torch
import numpy as np
import cv2

from config.data_config import DataConfig
from config.model_config import ModelConfig
from src.torch_utils.utils.misc import get_config_as_dict
from src.dataset.dataset_loaders import dog_vs_cat_loader as data_loader
from src.dataset.defeault_loader import default_load_data
from src.networks.build_network import build_model
from src.torch_utils.utils.draw import draw_pred_img
import src.dataset.data_transformations as transforms
from src.torch_utils.utils.misc import clean_print


def main():
    parser = ArgumentParser()
    parser.add_argument("model_path", help="Path to the checkpoint to use")
    parser.add_argument("data_path", help="Path to the test dataset")
    args = parser.parse_args()

    # Creates and load the model
    model = build_model(ModelConfig.MODEL, DataConfig.NB_CLASSES,
                        model_path=args.model_path, eval=True,  **get_config_as_dict(ModelConfig))
    print("Weights loaded", flush=True)

    data, labels, paths = data_loader(args.data_path, DataConfig.LABEL_MAP,
                                      data_preprocessing_fn=default_load_data, return_img_paths=True)
    base_cpu_pipeline = (transforms.resize(ModelConfig.IMAGE_SIZES), )
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
        cam = cv2.resize(cam, ModelConfig.IMAGE_SIZES)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        # Draw prediction (logits) on the image
        img = draw_pred(img, output, torch.Tensor([key]),
                        size=ModelConfig.IMAGE_SIZES, data_path=os.path.join(args.data_path, ".."))[0]

        # Fuse input image and gradcam mask
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap)
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        while True:
            cv2.imshow("Image", cam)
            if cv2.waitKey(10) == ord("q"):
                break


if __name__ == "__main__":
    main()
