from torch.utils.tensorboard import SummaryWriter  # noqa: F401  # Needs to be there to avoid segfaults
from argparse import ArgumentParser
from pathlib import Path
import shutil
import time
import sys
from functools import partial

import torch
from torchsummary import summary

from config.data_config import DataConfig
from config.model_config import ModelConfig
from src.torch_utils.utils.batch_generator import BatchGenerator
# from src.dataset.default_loader import default_loader as data_loader
from src.dataset.dataset_loaders import name_loader as data_loader
from src.dataset.default_loader import default_load_data
import src.dataset.data_transformations as transforms
from src.torch_utils.utils.misc import get_config_as_dict
from src.networks.build_network import build_model
from src.torch_utils.utils.misc import clean_print
from src.torch_utils.utils.logger import DummyLogger, create_logger
from src.train import train


def main():
    parser = ArgumentParser()
    parser.add_argument("--limit", "--l", default=None, type=int, help="Limits the number of apparition of each class")
    parser.add_argument("--load_data", "--ld", action="store_true", help="Loads all the images into RAM")
    parser.add_argument("--name", type=str, default="Train",
                        help="Used to know what a train is when using ps. Also name of the logger.")
    args = parser.parse_args()

    if not DataConfig.KEEP_TB:
        while DataConfig.TB_DIR.exists():
            shutil.rmtree(DataConfig.TB_DIR, ignore_errors=True)
            time.sleep(0.5)
    DataConfig.TB_DIR.mkdir(parents=True, exist_ok=True)

    if DataConfig.USE_CHECKPOINT:
        if not DataConfig.KEEP_CHECKPOINTS:
            while DataConfig.CHECKPOINT_DIR.exists():
                shutil.rmtree(DataConfig.CHECKPOINT_DIR, ignore_errors=True)
                time.sleep(0.5)
        try:
            DataConfig.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            print(f"The checkpoint dir {DataConfig.CHECKPOINT_DIR} already exists")
            return -1

        # Makes a copy of all the code (and config) so that the checkpoints are easy to load and use
        output_folder = DataConfig.CHECKPOINT_DIR / "Classification-PyTorch"
        for filepath in list(Path(".").glob("**/*.py")):
            destination_path = output_folder / filepath
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(filepath, destination_path)
        misc_files = ["README.md", "requirements.txt", "setup.cfg", ".gitignore"]
        for misc_file in misc_files:
            shutil.copy(misc_file, output_folder / misc_file)
        logger = create_logger(args.name, DataConfig.CHECKPOINT_DIR / "logs")
        print("Finished copying files")
    else:
        logger = DummyLogger()

    torch.backends.cudnn.benchmark = True   # Makes training quite a bit faster

    # Data augmentation done on cpu.
    base_cpu_pipeline = (
        transforms.resize(ModelConfig.IMAGE_SIZES),
    )
    cpu_augmentation_pipeline = transforms.compose_transformations((
        *base_cpu_pipeline,
        transforms.vertical_flip,
        transforms.horizontal_flip,
        transforms.rotate180,
        partial(transforms.rotate, min_angle=-20, max_angle=20),
        partial(transforms.cut_out, size=0.1)
    ))

    # GPU pipeline used by both validation and train
    base_gpu_pipeline = (
        transforms.to_tensor(),
        transforms.normalize_fn,
        # transforms.padding(ModelConfig.IMAGE_SIZES),
    )
    gpu_augmentation_pipeline = transforms.compose_transformations((
        *base_gpu_pipeline,
        transforms.noise()
    ))

    train_data, train_labels = data_loader(DataConfig.DATA_PATH / "Train", DataConfig.LABEL_MAP,
                                           limit=args.limit, load_data=args.load_data,
                                           data_preprocessing_fn=default_load_data if args.load_data else None)
    clean_print("Train data loaded")

    val_data, val_labels = data_loader(DataConfig.DATA_PATH / "Validation", DataConfig.LABEL_MAP,
                                       limit=args.limit, load_data=args.load_data,
                                       data_preprocessing_fn=default_load_data if args.load_data else None,
                                       shuffle=True)
    clean_print("Validation data loaded")
    print("Constructing dataloaders. . .", end="\r")

    with BatchGenerator(train_data, train_labels,
                        ModelConfig.BATCH_SIZE, nb_workers=DataConfig.NB_WORKERS,
                        data_preprocessing_fn=default_load_data if not args.load_data else None,
                        cpu_pipeline=cpu_augmentation_pipeline,
                        gpu_pipeline=gpu_augmentation_pipeline,
                        shuffle=True) as train_dataloader, \
        BatchGenerator(val_data, val_labels, ModelConfig.BATCH_SIZE, nb_workers=DataConfig.NB_WORKERS,
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
        model = build_model(ModelConfig.MODEL, DataConfig.NB_CLASSES, **get_config_as_dict(ModelConfig))
        summary(model, train_dataloader.data_shape)

        train(model, train_dataloader, val_dataloader, logger)


if __name__ == '__main__':
    main()
