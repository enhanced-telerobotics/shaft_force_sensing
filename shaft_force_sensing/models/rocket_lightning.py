"""PyTorch Lightning module for ROCKET model training."""

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter

from .rocket_models import (
    RocketForceRegressor,
    MultiRocketForceRegressor,
    HydraForceRegressor,
)


class LitRocketBase(pl.LightningModule):
    """Base Lightning wrapper for sktime RocketRegressor-based models.

    ROCKET models are one-shot sklearn-compatible estimators — they have
    no gradient loop, no optimizer, and no iterative training steps.
    This class wires them into Lightning only for the fit/predict
    lifecycle and shared logging conventions.

    Parameters
    ----------
    rocket_model : sklearn-compatible regressor
        An instance of RocketForceRegressor, MultiRocketForceRegressor,
        or HydraForceRegressor.
    """

    def __init__(self, rocket_model=None):
        super().__init__()
        self.rocket_model = rocket_model
        self._is_fitted = False
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        
        """Run inference using the fitted ROCKET model.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, n_channels, n_timepoints)

        Returns
        -------
        torch.Tensor
            Shape (batch, n_outputs)
        """

        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling forward()")

        predictions = self.rocket_model.predict(x.detach().cpu().numpy())

        # Ensure predictions is (n_samples, n_outputs)
        predictions = np.atleast_2d(predictions)
        if predictions.shape[0] != x.shape[0]:
            predictions = predictions.T

        return torch.tensor(predictions, dtype=x.dtype, device=x.device)    
        
            
    def training_step(self, batch, batch_idx):
        """Not used — ROCKET is a one-shot fit, not an iterative learner."""
        return None

    def configure_optimizers(self):
        """Not used — ROCKET has no gradient-based optimizer."""
        return None

    def train_rocket(self, train_loader, val_loader=None, writer=None):
        X_train_list, y_train_list = [], []
        for x, y, _ in train_loader:
            X_train_list.append(x.detach().cpu().numpy())
            y_train_list.append(y.detach().cpu().numpy())

        X_train = np.vstack(X_train_list).astype(np.float64)
        y_train = np.vstack(y_train_list).astype(np.float64)

        print(f"Training ROCKET model with {X_train.shape[0]} samples...")
        self.rocket_model.fit(X_train, y_train)
        self._is_fitted = True

        if val_loader is not None:
            X_val_list, y_val_list = [], []
            for x, y, _ in val_loader:
                X_val_list.append(x.detach().cpu().numpy())
                y_val_list.append(y.detach().cpu().numpy())

            X_val = np.vstack(X_val_list).astype(np.float64)
            y_val = np.vstack(y_val_list).astype(np.float64)

            y_pred = self.rocket_model.predict(X_val)

            # Ensure y_pred is (n_samples, n_outputs)
            y_pred = np.atleast_2d(y_pred)
            if y_pred.shape[0] != X_val.shape[0]:
                y_pred = y_pred.T

            val_loss = self.loss_fn(
                torch.tensor(y_pred, dtype=torch.float32),
                torch.tensor(y_val,  dtype=torch.float32),
            )
            print(f"Validation MSE Loss: {val_loss.item():.4f}")
            if writer is not None:
                writer.add_scalar("val/loss", val_loss.item(), global_step=0)

class LitRocket(LitRocketBase):
    """Lightning wrapper for RocketForceRegressor.

    Parameters
    ----------
    num_kernels : int, optional
        Number of random convolutional kernels (default: 10000).
    random_state : int, optional
        Random seed (default: None).
    n_jobs : int, optional
        Parallel jobs (default: 1).
    """

    def __init__(self, num_kernels=10000, random_state=None, n_jobs=1):
        super().__init__(rocket_model=RocketForceRegressor(
            num_kernels=num_kernels,
            random_state=random_state,
            n_jobs=n_jobs,
        ))


class LitMultiRocket(LitRocketBase):
    """Lightning wrapper for MultiRocketForceRegressor.

    Parameters
    ----------
    num_kernels : int, optional
        Number of kernels (default: 6250). Rounded down to nearest
        multiple of 84 internally.
    random_state : int, optional
        Random seed (default: None).
    n_jobs : int, optional
        Parallel jobs (default: 1).
    """

    def __init__(self, num_kernels=6250, random_state=None, n_jobs=1):
        super().__init__(rocket_model=MultiRocketForceRegressor(
            num_kernels=num_kernels,
            random_state=random_state,
            n_jobs=n_jobs,
        ))


class LitHydra(LitRocketBase):
    """Lightning wrapper for HydraForceRegressor.

    NOTE: sktime has no native Hydra transformer. HydraForceRegressor
    uses MultiRocket (multirocket + use_multivariate='yes') as the
    closest sktime substitute. For true Hydra use aeon's HydraTransformer
    or https://github.com/angus924/hydra

    Parameters
    ----------
    num_kernels : int, optional
        Number of kernels (default: 6250). Rounded down to nearest
        multiple of 84 internally.
    random_state : int, optional
        Random seed (default: None).
    n_jobs : int, optional
        Parallel jobs (default: 1).
    """

    def __init__(self, num_kernels=6250, random_state=None, n_jobs=1):
        super().__init__(rocket_model=HydraForceRegressor(
            num_kernels=num_kernels,
            random_state=random_state,
            n_jobs=n_jobs,
        ))