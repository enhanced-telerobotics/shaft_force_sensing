"""PyTorch Lightning wrappers for ROCKET-based models."""

import torch
import numpy as np
import pytorch_lightning as pl

from .rocket_models import (
    RocketForceRegressor,
    MultiRocketForceRegressor,
    HydraForceRegressor,
)
from .lightning import LitSequenceModel


class LitRocketBase(LitSequenceModel):
    """Base class for PyTorch Lightning ROCKET model wrappers.
    
    This class wraps scikit-learn compatible ROCKET models from aeon-toolkit
    into the PyTorch Lightning framework for training and evaluation.
    """
    
    def __init__(self, rocket_model=None, **kwargs):
        """Initialize the Lightning ROCKET wrapper.
        
        Parameters
        ----------
        rocket_model : sklearn-compatible regressor
            An instance of a ROCKET regressor
        **kwargs : dict
            Additional arguments for LitSequenceModel
        """
        super().__init__(**kwargs)
        self.rocket_model = rocket_model
        self._is_fitted = False
    
    def forward(self, x, *args, **kwargs):
        """Forward pass using the fitted ROCKET model.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, features)
        
        Returns
        -------
        torch.Tensor
            Predicted force values of shape (batch, d_output)
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling forward()")
        
        # Convert to numpy
        x_np = x.detach().cpu().numpy()
        
        # Make predictions
        predictions = self.rocket_model.predict(x_np)
        
        # Convert back to torch tensor on the same device as input
        return torch.tensor(predictions, dtype=x.dtype, device=x.device)
    
    def training_step(self, batch, batch_idx):
        """Training step - not used since ROCKET is not iterative."""
        # For ROCKET models, training happens in train_rocket()
        # This is kept for API compatibility
        return None
    
    def on_train_epoch_start(self):
        """Called at the start of training epoch."""
        pass
    
    def train_rocket(self, train_loader, val_loader=None):
        """Train the ROCKET model on provided data.
        
        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader, optional
            Validation data loader for early stopping (not used for ROCKET)
        """
        # Collect all training data
        X_train_list = []
        y_train_list = []
        
        for batch in train_loader:
            x, y, _ = batch
            X_train_list.append(x.detach().cpu().numpy())
            y_train_list.append(y.detach().cpu().numpy())
        
        X_train = np.vstack(X_train_list)
        y_train = np.vstack(y_train_list).squeeze()
        
        # Fit the model
        print(f"Training ROCKET model with {X_train.shape[0]} samples...")
        self.rocket_model.fit(X_train, y_train)
        self._is_fitted = True
        
        # Evaluate on validation set if provided
        if val_loader is not None:
            X_val_list = []
            y_val_list = []
            
            for batch in val_loader:
                x, y, _ = batch
                X_val_list.append(x.detach().cpu().numpy())
                y_val_list.append(y.detach().cpu().numpy())
            
            X_val = np.vstack(X_val_list)
            y_val = np.vstack(y_val_list).squeeze()
            
            # Calculate validation metrics
            y_pred = self.rocket_model.predict(X_val)
            val_loss = self.loss_fn(
                torch.tensor(y_pred, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32)
            )
            print(f"Validation MSE Loss: {val_loss.item():.4f}")
            
            self.log("val/loss", val_loss, prog_bar=True, logger=True)


class LitRocket(LitRocketBase):
    """PyTorch Lightning wrapper for RocketRegressor.
    
    Parameters
    ----------
    num_kernels : int, optional
        Number of random convolutional kernels (default: 10000)
    **kwargs : dict
        Additional arguments for LitSequenceModel
    """
    
    def __init__(self, num_kernels=10000, **kwargs):
        """Initialize LitRocket wrapper."""
        rocket_model = RocketForceRegressor(num_kernels=num_kernels)
        super().__init__(rocket_model=rocket_model, **kwargs)


class LitMultiRocket(LitRocketBase):
    """PyTorch Lightning wrapper for MultiRocketRegressor.
    
    MultiROCKET uses multiple scales of random kernels for better
    temporal pattern capture.
    
    Parameters
    ----------
    num_kernels : int, optional
        Number of random convolutional kernels (default: 5000)
    **kwargs : dict
        Additional arguments for LitSequenceModel
    """
    
    def __init__(self, num_kernels=5000, **kwargs):
        """Initialize LitMultiRocket wrapper."""
        rocket_model = MultiRocketForceRegressor(num_kernels=num_kernels)
        super().__init__(rocket_model=rocket_model, **kwargs)


class LitHydra(LitRocketBase):
    """PyTorch Lightning wrapper for HydraRegressor.
    
    Hydra combines ROCKET with ensemble Ridge regressors at different
    regularization strengths.
    
    Parameters
    ----------
    num_kernels : int, optional
        Number of random convolutional kernels (default: 2000)
    **kwargs : dict
        Additional arguments for LitSequenceModel
    """
    
    def __init__(self, num_kernels=2000, **kwargs):
        """Initialize LitHydra wrapper."""
        rocket_model = HydraForceRegressor(num_kernels=num_kernels)
        super().__init__(rocket_model=rocket_model, **kwargs)
