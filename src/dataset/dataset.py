import torch
import cv2
import numpy as np

from src.dataset.dataset_utils import default_loader
from config.data_config import DataConfig


class Dataset(torch.utils.data.Dataset):
    """Classification dataset."""

    def __init__(self, data_path: str, transform=None, limit: int = None, load_images: bool = True):
        """
        Args:
            data_path:
                Path to the root folder of the dataset.
                This folder is expected to contain subfolders for each class, with the images inside.
                It should also contain a "class.names" with all the classes
            transform (callable, optional): Optional transform to be applied on a sample.
            limit (int, optional): If given then the number of elements for each class in the dataset
                                   will be capped to this number
            load_images: If True then all the images are loaded into ram
        """
        self.transform = transform
        self.load_images = load_images

        self.labels = default_loader(data_path, DataConfig.LABEL_MAP, limit=limit, load_images=load_images)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.tolist()

        if self.load_images:
            img = self.labels[i, 0].astype(np.uint8)
        else:
            img = cv2.imread(self.labels[i, 0])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = int(self.labels[i, 1])
        sample = {'img': img, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
