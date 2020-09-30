import argparse
import glob
import os

import cv2
import torch
from torchvision.transforms import Compose
import numpy as np

from config.data_config import DataConfig
from config.model_config import ModelConfig
from src.networks.build_network import build_model
from src.utils.draw import draw_pred
import src.dataset.transforms as transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to the checkpoint to use")
    parser.add_argument("data_path", help="Path to the test dataset")
    args = parser.parse_args()

    # Creates and load the model
    model = build_model(ModelConfig.NETWORK, args.model_path, eval=True)
    print("Weights loaded", flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    label_map = DataConfig.LABEL_MAP
    transform = Compose([
        transforms.Crop(top=600, bottom=500, left=800, right=200),
        transforms.Resize(*ModelConfig.IMAGE_SIZES),
        transforms.Normalize(),
        transforms.ToTensor()
    ])

    img_types = ("*.jpg", "*.bmp")
    for key in range(len(label_map)):
        pathname = os.path.join(args.data_path, label_map[key], "**")
        image_paths = []
        [image_paths.extend(glob.glob(os.path.join(pathname, ext), recursive=True)) for ext in img_types]
        for img_path in image_paths:
            msg = f"Loading data {img_path}"
            print(msg + ' ' * (os.get_terminal_size()[0] - len(msg)), end="\r")
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = transform({"img": img, "label": 0})["img"]   # The 0 is ignored
            img = img.unsqueeze(0).to(device).float()

            # Feed it to the model
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
