import os

import torch
import cv2
import numpy as np

from src.dataset.dataset_utils import dogs_vs_cats


class Dataset(torch.utils.data.Dataset):
    """Classification dataset."""

    def __init__(self, data_path: str, transform=None):
        """
        Args:
            data_path:
                Path to the root folder of the dataset.
                This folder is expected to contain subfolders for each class, with the images inside.
                It should also contain a "class.names" with all the classes
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform

        # Build a map between id and names
        self.label_map = {}
        with open(os.path.join(data_path, "..", "classes.names")) as table_file:
            for key, line in enumerate(table_file):
                label = line.strip()
                self.label_map[key] = label

        self.labels = dogs_vs_cats(data_path, self.label_map)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.tolist()

        img = cv2.imread(self.labels[i, 0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.labels[i, 1].astype(np.uint8)
        sample = {'img': img, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
