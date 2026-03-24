from __future__ import annotations
from typing import Iterator, Tuple
import numpy as np
import mytorch
from .dataset import Dataset
from ..tensor import Tensor

class DataLoader():
    def __init__(self : DataLoader, dataset : Dataset, batch_size : int, device : str):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self._counter = 0

    def __len__(self : DataLoader):
        return int(len(self.dataset) / self.batch_size)

    def __iter__(self : DataLoader) -> Iterator:
        self._counter = 0
        return self
    
    def __next__(self : DataLoader) -> Tuple[Tensor, Tensor]:
        if self._counter == len(self):
            raise StopIteration
        
        start_idx = self._counter * self.batch_size
        end_idx = np.minimum((self._counter + 1) * self.batch_size, len(self.dataset))
        idx = range(start_idx, end_idx)
        
        X, y = self.dataset[idx]
        X_batch = mytorch.Tensor(X, device=self.device)
        y_batch = mytorch.Tensor(y, device=self.device, dtype="int32")

        self._counter += 1
        return X_batch, y_batch