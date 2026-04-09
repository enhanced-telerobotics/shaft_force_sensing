"""Dataset classes for force sensing models."""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler

JOINTS = [
    'yaw',
    'pitch',
    'insertion',
    'roll',
    'wrist_pitch',
    'wrist_yaw',
    'jaw'
]


class SensorDataset(Dataset):
    """Dataset for force sensing using shaft sensor. 

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

class TorqueDataset(Dataset):
    """Dataset for internal torque sensing methods. 
    
    """
    def __init__(self,
                 data_path: Path,
                 stride: int = 1,
                 sequence_length: int = 1000,
                 joints: int = 6):
        self.sequence_length = sequence_length

        # Load data
        data = pd.read_csv(data_path)

        # Downsample by stride
        self.indices = np.arange(0, len(data), stride)

        # Select columns
        self.joints = JOINTS[:joints]
        p_cols = [f'{joint}_position' for joint in self.joints]
        v_cols = [f'{joint}_velocity' for joint in self.joints]
        t_cols = [f'{joint}_effort' for joint in self.joints]
        f_cols = ['ati_fx', 'ati_fy', 'ati_fz']

        # Split data
        self.positions = data[p_cols].to_numpy()
        self.velocities = data[v_cols].to_numpy()
        self.torques = data[t_cols].to_numpy()
        self.forces = data[f_cols].to_numpy()

        # Load Jacobians and rotaions
        data = np.load(data_path.with_suffix('.npz'))
        self.jaco = data['jacobians']
        self.rot = data['rotations']

    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Returns:
            pos_seq (torch.Tensor): Joint position sequence of shape (sequence_length, joints)
            vel_seq (torch.Tensor): Joint velocity sequence of shape (sequence_length, joints)
            torque (torch.Tensor): Joint torque at current timestep of shape (joints, )
            force_target (torch.Tensor): Force target of shape (3, )
            jaco (torch.Tensor): Jacobian matrix at current timestep of shape (6, joints)
            seq_len (torch.Tensor): Number of valid (non-padded) timesteps, shape ()
        """
        data_idx = self.indices[idx]
        start = data_idx - self.sequence_length + 1
        valid_start = max(0, start)

        pos_seq = self.positions[valid_start:data_idx + 1]
        vel_seq = self.velocities[valid_start:data_idx + 1]
        torque = self.torques[data_idx]
        force_target = self.forces[data_idx]
        jaco = self.jaco[data_idx]

        seq_len = pos_seq.shape[0]

        # Right-pad with zeros so valid timesteps are in [0, seq_len).
        if seq_len < self.sequence_length:
            pad_right = self.sequence_length - seq_len
            pos_pad = np.zeros((pad_right, pos_seq.shape[1]), dtype=pos_seq.dtype)
            vel_pad = np.zeros((pad_right, vel_seq.shape[1]), dtype=vel_seq.dtype)
            pos_seq = np.vstack([pos_seq, pos_pad])
            vel_seq = np.vstack([vel_seq, vel_pad])
        elif seq_len > self.sequence_length:
            pos_seq = pos_seq[-self.sequence_length:]
            vel_seq = vel_seq[-self.sequence_length:]
            seq_len = self.sequence_length

        return (
            torch.as_tensor(pos_seq, dtype=torch.float32),
            torch.as_tensor(vel_seq, dtype=torch.float32),
            torch.as_tensor(torque, dtype=torch.float32),
            torch.as_tensor(force_target, dtype=torch.float32),
            torch.as_tensor(jaco, dtype=torch.float32),
            torch.as_tensor(seq_len, dtype=torch.long),
        )