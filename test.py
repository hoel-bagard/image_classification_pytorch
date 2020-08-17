import argparse
import glob
import os

import torch
import numpy as np
import cv2

from config.data_config import DataConfig
from src.networks.small_darknet import SmallDarknet
from src.utils.draw import draw_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to the checkpoint to use")
    parser.add_argument("data_path", help="Path to the test dataset")
    parser.add_argument("--show", action="store_true", help="Show the bad images")
    args = parser.parse_args()

    # Creates and load the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SmallDarknet()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    model.to(device)
    print("Weights loaded", flush=True)

    label_map = {}
    with open(os.path.join(args.data_path, "..", "class.names")) as table_file:
        for key, line in enumerate(table_file):
            label = line.strip()
            label_map[key] = label

    results = []
    for key in range(len(label_map)):
        for img_path in glob.glob(os.path.join(args.data_path, f"{label_map[key]}*.jpg")):
            img = cv2.imread(img_path)

            img = np.array(img)[:, :, :3]/255
            img = cv2.resize(img, (DataConfig.SIZE, DataConfig.SIZE))
            img = img.transpose((2, 0, 1))
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(device).float()

            output = model(img)
            prediction = np.argmax(output.cpu().detach().numpy())
            if key == prediction:
                results.append(1)
            else:
                results.append(0)

            if args.show and key != prediction:
                out_img = draw_pred(torch.Tensor([img]), torch.Tensor([output]), torch.Tensor([key]))[0]
                cv2.imshow("Image", out_img[0])
                cv2.waitKey()

    results = np.asarray(results)
    print(f"Precision: {np.mean(results)}")


if __name__ == "__main__":
    main()
