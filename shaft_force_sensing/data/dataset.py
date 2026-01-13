"""Dataset classes for force sensing models."""

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    """Sliding window dataset for time series force prediction.
    
    Parameters
    ----------
    inputs : np.ndarray
        Input features of shape [T, input_size]
    force_targets : np.ndarray
        Target force values of shape [T, 3]
    lookback : int, optional
        Number of timesteps of history for each sample (default: 150)
    stride : int, optional
        Step size when moving the window forward (default: 3)
    normalize : bool, optional
        Whether to normalize targets (default: True)
    f_mean : np.ndarray, optional
        Pre-computed mean for normalization
    f_std : np.ndarray, optional
        Pre-computed std for normalization
    """

    def __init__(
        self,
        inputs,
        force_targets,
        lookback=150,
        stride=3,
        normalize=True,
        f_mean=None,
        f_std=None,
    ):
        """Initialize the dataset."""
        self.inputs = inputs
        self.lookback = lookback
        self.stride = stride

        # Compute stats for normalization
        if normalize:
            if f_mean is None and f_std is None:
                self.force_mean = force_targets.mean(axis=0)
                self.force_std = force_targets.std(axis=0) + 1e-8
            else:
                self.force_mean = f_mean
                self.force_std = f_std

            self.force_targets = (force_targets - self.force_mean) / self.force_std
        else:
            self.force_mean = np.zeros(force_targets.shape[1])
            self.force_std = np.ones(force_targets.shape[1])
            self.force_targets = force_targets

        # Number of valid windows
        self.length = (inputs.shape[0] - lookback) // stride + 1

    def __len__(self):
        """Return dataset length."""
        return self.length

    def __getitem__(self, idx):
        """Get a sample from the dataset.
        
        Returns
        -------
        tuple
            (input_tensor, target_tensor) where input_tensor is [lookback, input_size]
            and target_tensor is [3]
        """
        start = idx * self.stride
        end = start + self.lookback

        x = self.inputs[start:end]  # [lookback, input_size]
        force_y = self.force_targets[end - 1]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(force_y, dtype=torch.float32),
        )
