import argparse
import glob
import os

import torch
import numpy as np
import cv2

from config.data_config import DataConfig
from src.networks.auto_encoder import AutoEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to the checkpoint to use")
    parser.add_argument("data_path", help="Path to the test dataset")
    parser.add_argument("--show", action="store_true", help="Show the bad images")
    args = parser.parse_args()

    # Creates and load the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoEncoder()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    model.to(device)
    print("Weights loaded", flush=True)

    dataset_paths = [os.path.join(args.data_path, "Good"),
                     os.path.join(args.data_path, "Bad")]
    data_cls = ["Good", "Bad"]

    results = []
    for data_path, cls in zip(dataset_paths, data_cls):
        for img_path in glob.glob(os.path.join(data_path, "*")):
            img = cv2.imread(img_path)

            img = np.array(img)[:, :, :3]/255
            img = cv2.resize(img, (DataConfig.SIZE, DataConfig.SIZE))
            img = img.transpose((2, 0, 1))
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(device).float()

            output = model(img)
            error = torch.mean((img - output) ** 2).cpu().detach().numpy()
            results.append((cls, error))

            if args.show and cls == "Bad":
                cv2.imshow("Input image", img.cpu().detach().numpy()[0].transpose(1, 2, 0))
                cv2.imshow("Output image", output.cpu().detach().numpy()[0].transpose(1, 2, 0))
                cv2.waitKey()

    results = np.asarray(results)
    errors = np.asarray(results[:, 1], dtype=np.float32)
    classes = np.asarray(results[:, 0], dtype=np.str)
    mean = np.mean(errors)
    std = np.std(errors)
    thresh = mean
    print(f"Mean: {mean:.5e}, std: {std:.5e}")
    tp, fp, tn, fn = 0, 0, 0, 0
    for cls, error in zip(classes, errors):
        if cls == "Bad" and error > thresh:
            tp += 1
        elif cls == "Good" and error > thresh:
            fp += 1
        if cls == "Good" and error < thresh:
            tn += 1
        elif cls == "Bad" and error < thresh:
            fn += 1

    print(f"Confusion matrix: {[[tp, fp], [fn, tn]]}")
    print(f"Accuracy: {(tp+tn) / (tp + tn+ fp + fn)}")
    print(f"Precision: {tp / (tp + fp)}")
    print(f"Recall: {tp / (tp + fn)}")
    print(f"F1 score: {tp / (tp + 0.5*(fp + fn))}")


if __name__ == "__main__":
    main()
