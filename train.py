import argparse
import shutil
import sys
import time
from functools import partial
from pathlib import Path

import torch

import src.dataset.data_transformations as transforms
from config.data_config import get_data_config
from config.model_config import get_model_config
from src.dataset.default_loader import default_load_data
from src.dataset.default_loader import default_loader as data_loader
# from src.dataset.dataset_loaders import name_loader as data_loader
from src.networks.build_network import build_model
from src.torch_utils.utils.batch_generator import BatchGenerator
from src.torch_utils.utils.logger import create_logger
from src.torch_utils.utils.misc import clean_print
from src.torch_utils.utils.misc import get_config_as_dict
from src.torch_utils.utils.prepare_folders import prepare_folders
from src.torch_utils.utils.torch_summary import summary
from src.train import train


def main():
    parser = argparse.ArgumentParser(description="Training script",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--limit", "--l", default=None, type=int, help="Limits the number of apparition of each class")
    parser.add_argument("--load_data", "--ld", action="store_true", help="Loads all the images into RAM")
    parser.add_argument("--name", type=str, default="Train",
                        help="Used to know what a train is when using ps. Also name of the logger.")
    parser.add_argument("--verbose_level", "-v", choices=["debug", "info", "error"], default="info", type=str,
                        help="Logger level.")
    args = parser.parse_args()

    limit: int = args.limit
    name: str = args.name
    verbose_level: str = args.verbose_level

    data_config = get_data_config()
    model_config = get_model_config()

    log_dir = data_config.CHECKPOINTS_DIR / "print_logs" if data_config.USE_CHECKPOINTS else None
    logger = create_logger(name, log_dir=log_dir, verbose_level=verbose_level)

    if not data_config.KEEP_TB:
        while data_config.TB_DIR.exists():
            shutil.rmtree(data_config.TB_DIR, ignore_errors=True)
            time.sleep(0.5)
    data_config.TB_DIR.mkdir(parents=True, exist_ok=True)

    if data_config.USE_CHECKPOINT:
        if not data_config.KEEP_CHECKPOINTS:
            while data_config.CHECKPOINT_DIR.exists():
                shutil.rmtree(data_config.CHECKPOINT_DIR, ignore_errors=True)
                time.sleep(0.5)
        try:
            data_config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            print(f"The checkpoint dir {data_config.CHECKPOINT_DIR} already exists")
            return -1

        # Makes a copy of all the code (and config) so that the checkpoints are easy to load and use
        output_folder = data_config.CHECKPOINT_DIR / "Classification-PyTorch"
        for filepath in list(Path(".").glob("**/*.py")):
            destination_path = output_folder / filepath
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(filepath, destination_path)
        misc_files = ["README.md", "requirements.txt", "setup.cfg", ".gitignore"]
        for misc_file in misc_files:
            shutil.copy(misc_file, output_folder / misc_file)
        print("Finished copying files")
    logger.info("Finished preparing tensorboard and checkpoints folders.")

    torch.backends.cudnn.benchmark = True   # Makes training quite a bit faster

    # Data augmentation done on cpu.
    base_cpu_pipeline = (
        # transforms.resize(model_config.IMAGE_SIZES),
    )
    cpu_augmentation_pipeline = transforms.compose_transformations((
        *base_cpu_pipeline,
        transforms.vertical_flip,
        transforms.horizontal_flip,
        transforms.rotate180,
        transforms.rotate90,
        transforms.temp_pil_aug,
        partial(transforms.rotate, min_angle=-10, max_angle=10),
        partial(transforms.cut_out, size=0.15)
    ))

    # GPU pipeline used by both validation and train
    base_gpu_pipeline = (
        transforms.to_tensor(),
        transforms.normalize_fn,
        transforms.padding(model_config.IMAGE_SIZES),
    )
    gpu_augmentation_pipeline = transforms.compose_transformations((
        *base_gpu_pipeline,
        transforms.noise()
    ))

    train_data, train_labels = data_loader(data_config.DATA_PATH / "Train", data_config.LABEL_MAP,
                                           limit=args.limit, load_data=args.load_data,
                                           data_preprocessing_fn=default_load_data if args.load_data else None)
    clean_print("Train data loaded")

    val_data, val_labels = data_loader(data_config.DATA_PATH / "Validation", data_config.LABEL_MAP,
                                       limit=args.limit, load_data=args.load_data,
                                       data_preprocessing_fn=default_load_data if args.load_data else None,
                                       shuffle=True)
    clean_print("Validation data loaded")
    print("Constructing dataloaders. . .", end="\r")

    with BatchGenerator(train_data, train_labels,
                        model_config.BATCH_SIZE, nb_workers=data_config.NB_WORKERS,
                        data_preprocessing_fn=default_load_data if not args.load_data else None,
                        cpu_pipeline=cpu_augmentation_pipeline,
                        gpu_pipeline=gpu_augmentation_pipeline,
                        shuffle=True) as train_dataloader, \
        BatchGenerator(val_data, val_labels, model_config.BATCH_SIZE, nb_workers=data_config.NB_WORKERS,
                       data_preprocessing_fn=default_load_data if not args.load_data else None,
                       cpu_pipeline=transforms.compose_transformations(base_cpu_pipeline),
                       gpu_pipeline=transforms.compose_transformations(base_gpu_pipeline),
                       shuffle=False) as val_dataloader:

        clean_print("                               \n")
        logger.info("-------- Starting train --------")
        logger.info("From command : " + ' '.join(sys.argv))
        logger.info(f"Loaded {len(train_dataloader)} train data and "
                    f"{len(val_dataloader)} validation data\n")

        print("Building model. . .", end="\r")
        model = build_model(model_config.MODEL, data_config.NB_CLASSES, **get_config_as_dict(model_config))
        summary(model, train_dataloader.data_shape)

        train(model, train_dataloader, val_dataloader, logger)


if __name__ == '__main__':
    main()
