import time
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch
from config.record_config import DataConfig
from config.train_config import ModelConfig

import classification.data.data_transformations as transforms
from classification.data.dataset_loaders import dog_vs_cat_loader as data_loader
from classification.data.default_loader import default_load_data
from classification.networks.build_network import build_model
from classification.torch_utils.utils.draw import draw_pred_img
from classification.torch_utils.utils.misc import clean_print, get_config_as_dict


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("model_path", type=Path, help="Path to the checkpoint to use")
    parser.add_argument("data_path", type=Path, help="Path to the test dataset")
    parser.add_argument("--show", "--s", action="store_true", help="Show the images where the network failed.")
    args = parser.parse_args()

    inference_start_time = time.perf_counter()

    # Creates and load the model
    model = build_model(
        ModelConfig.MODEL,
        DataConfig.NB_CLASSES,
        model_path=args.model_path,
        eval=True,
        **get_config_as_dict(ModelConfig),
    )
    print("Weights loaded", flush=True)

    data, labels, paths = data_loader(
        args.data_path, DataConfig.LABEL_MAP, data_preprocessing_fn=default_load_data, return_img_paths=True
    )
    base_cpu_pipeline = (transforms.resize(ModelConfig.IMAGE_SIZES),)
    base_gpu_pipeline = (transforms.to_tensor(), transforms.normalize(labels_too=True))
    data_transformations = transforms.compose_transformations((*base_cpu_pipeline, *base_gpu_pipeline))
    print("\nData loaded", flush=True)

    results = []  # Variable used to keep track of the classification results
    for img, label, img_path in zip(data, labels, paths):
        clean_print(f"Processing image {img_path}", end="\r")
        img, label = data_transformations([img], [label])
        with torch.no_grad():
            output = model(img)
            output = torch.nn.functional.softmax(output, dim=-1)
            prediction = torch.argmax(output)
            pred_correct = label == prediction
            if pred_correct:
                results.append(1)
            else:
                results.append(0)

            if args.show and not pred_correct:
                out_img = draw_pred_img(img, output, label, DataConfig.LABEL_MAP, size=ModelConfig.IMAGE_SIZES)
                out_img = cv2.cvtColor(out_img[0], cv2.COLOR_RGB2BGR)
                while True:
                    cv2.imshow("Image", out_img)
                    key = cv2.waitKey(10)
                    if key == ord("q"):
                        cv2.destroyAllWindows()
                        break

    results = np.asarray(results)
    total_time = time.perf_counter() - inference_start_time
    print("\nFinished running inference on the test dataset.")
    print(f"Total inference time was {total_time:.3f}s, which averages to {total_time/len(results):.5f}s per image")
    print(f"Precision: {np.mean(results)}")


if __name__ == "__main__":
    main()
