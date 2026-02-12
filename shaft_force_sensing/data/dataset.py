"""Dataset classes for force sensing models."""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler


class ForceSensingDataset(Dataset):
    """Dataset for shaft force sensing. 

    Args:
        data_path (Path): Path to the CSV data file.
        stride (int): Stride for downsampling the data.
        sequence_length (int): Length of input sequences for the model.
        nomalizer (StandardScaler, optional): Pre-fitted scaler for normalizing targets. If None, a new scaler will be fitted on the data.
    """

    def __init__(self,
                 data_path: Path,
                 input_cols: list,
                 target_cols: list,
                 stride: int = 1,
                 sequence_length: int = 100,
                 nomalizer: StandardScaler = None):
        self.sequence_length = sequence_length

        # Load data
        data = pd.read_csv(data_path)

        # Downsample by stride
        self.indices = np.arange(0, len(data), stride)

        # Split into input and target
        self.X = data[input_cols].to_numpy()
        self.y = data[target_cols].to_numpy()

        # Normalize targets
        if nomalizer is not None:
            self.y = nomalizer.transform(self.y)
        else:
            nomalizer = StandardScaler()
            self.y = nomalizer.fit_transform(self.y)

        self.nomalizer = nomalizer

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Returns:
            X_seq (torch.Tensor): Input sequence of shape (sequence_length, input_size)
            y (torch.Tensor): Target of shape (target_size, )
            mask (torch.Tensor): Mask indicating valid data points (True for padding, False for valid data) of shape (sequence_length,)
        """
        data_idx = self.indices[idx]
        start = data_idx - self.sequence_length + 1

        if start < 0:
            pad_left = -start
            valid_start = 0
        else:
            pad_left = 0
            valid_start = start

        X_seq = self.X[valid_start:data_idx + 1]
        y_target = self.y[data_idx]

        if pad_left > 0:
            X_pad = np.zeros((pad_left, X_seq.shape[1]), dtype=X_seq.dtype)
            X_seq = np.vstack([X_pad, X_seq])

        if X_seq.shape[0] != self.sequence_length:
            X_seq = X_seq[-self.sequence_length:]
            pad_left = max(0, self.sequence_length - (data_idx + 1))

        mask = np.zeros(self.sequence_length, dtype=bool)
        if pad_left > 0:
            mask[:pad_left] = True

        return (
            torch.as_tensor(X_seq, dtype=torch.float32),
            torch.as_tensor(y_target, dtype=torch.float32),
            torch.as_tensor(mask, dtype=torch.bool),
        )
