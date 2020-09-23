import argparse
import glob
import os

import cv2
import torch
import numpy as np

from config.model_config import ModelConfig
from src.networks.small_darknet import SmallDarknet
from src.networks.wide_net import WideNet
from src.utils.draw import draw_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to the checkpoint to use")
    parser.add_argument("data_path", help="Path to the test dataset")
    parser.add_argument("--show", action="store_true", help="Show the bad images")
    args = parser.parse_args()

    # Creates and load the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if ModelConfig.NETWORK == "SmallDarknet":
        model = SmallDarknet()
    elif ModelConfig.NETWORK == "WideNet":
        model = WideNet()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    model.to(device)
    print("Weights loaded", flush=True)

    label_map = {}
    with open(os.path.join(args.data_path, "..", "classes.names")) as table_file:
        for key, line in enumerate(table_file):
            label = line.strip()
            label_map[key] = label

    for key in range(len(label_map)):
        for img_path in glob.glob(os.path.join(args.data_path, f"{label_map[key]}*.jpg")):
            # Read and prepare image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.array(img)[:, :, :3]/255.0
            img = cv2.resize(img, ModelConfig.IMAGE_SIZES)
            img = img.transpose((2, 0, 1))
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(device).float()

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
