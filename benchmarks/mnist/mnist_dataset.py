import os
import gzip
import numpy as np
from dataset import Dataset

class MnistDatset(Dataset):
    def __init__(self, dataset_folder: str, train : bool):
        if train:
            data = os.path.join(dataset_folder, "train-images-idx3-ubyte.gz")
            labels = os.path.join(dataset_folder, "train-labels-idx1-ubyte.gz")
        else:
            data = os.path.join(dataset_folder, "t10k-images-idx3-ubyte.gz")
            labels = os.path.join(dataset_folder, "t10k-labels-idx1-ubyte.gz")

        self.X = self._parse_images(data)
        self.y = self._parse_labels(labels)

    @staticmethod
    def _parse_images(filename):
        with gzip.open(filename, "rb") as file:
            data = np.frombuffer(file.read(), np.uint8, offset = 16)
            return data.reshape(-1, 28, 28) / 255

    @staticmethod
    def _parse_labels(filename):
        with gzip.open(filename, "rb") as file:
            return np.frombuffer(file.read(), np.uint8, offset = 8)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        return X, y