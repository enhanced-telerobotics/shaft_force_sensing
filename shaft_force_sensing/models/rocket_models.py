"""ROCKET-based models for force prediction using aeon-toolkit.

ROCKET (Random Convolutional Kernel Transform) models are efficient
time series classification and regression methods that use random
convolutional kernels to transform input sequences.

References:
    - RocketRegressor: https://www.aeon-toolkit.org/en/stable/api_reference/regression.html#rocketsregressor
    - MultiRocketRegressor: https://www.aeon-toolkit.org/en/stable/api_reference/regression.html#multirocketregressor
    - HydraRegressor: https://www.aeon-toolkit.org/en/stable/api_reference/regression.html#hydraregressor
"""

from aeon.regression.rocket import (
    RocketRegressor,
    MultiRocketRegressor,
    HydraRegressor,
)
from sklearn.linear_model import Ridge
import numpy as np


class RocketForceRegressor(RocketRegressor):
    """ROCKET-based force prediction model.
    
    Uses random convolutional kernels to extract features from
    time series sequences for force regression.
    
    Parameters
    ----------
    num_kernels : int, optional
        Number of random convolutional kernels (default: 10000)
    normalise : bool, optional
        Whether to normalize inputs (default: True)
    random_state : int, optional
        Random seed for reproducibility (default: None)
    n_jobs : int, optional
        Number of parallel jobs (default: -1 for all processors)
    """
    
    def __init__(
        self,
        num_kernels=10000,
        normalise=True,
        random_state=None,
        n_jobs=-1,
    ):
        """Initialize ROCKET regressor."""
        super().__init__(
            num_kernels=num_kernels,
            normalise=normalise,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.d_output = 3  # For 3D force: fx, fy, fz
    
    def fit(self, X, y):
        """Fit the model to training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_channels, n_timepoints)
            Training input sequences
        y : array-like of shape (n_samples, n_outputs)
            Target force values
        
        Returns
        -------
        self : object
            Returns self
        """
        # Reshape if needed: (batch, seq_len, features) -> (batch, features, seq_len)
        if X.ndim == 3 and X.shape[2] > X.shape[1]:
            X = np.transpose(X, (0, 2, 1))
        
        return super().fit(X, y)
    
    def predict(self, X):
        """Predict force values.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_channels, n_timepoints)
            Input sequences
        
        Returns
        -------
        y_pred : array-like of shape (n_samples, n_outputs)
            Predicted force values
        """
        # Reshape if needed: (batch, seq_len, features) -> (batch, features, seq_len)
        if X.ndim == 3 and X.shape[2] > X.shape[1]:
            X = np.transpose(X, (0, 2, 1))
        
        return super().predict(X)


class MultiRocketForceRegressor(MultiRocketRegressor):
    """Multi-Scale ROCKET-based force prediction model.
    
    MultiROCKET uses random kernels at multiple scales to better
    capture temporal patterns at different granularities.
    
    Parameters
    ----------
    num_kernels : int, optional
        Number of random convolutional kernels (default: 5000)
    normalise : bool, optional
        Whether to normalize inputs (default: True)
    random_state : int, optional
        Random seed for reproducibility (default: None)
    n_jobs : int, optional
        Number of parallel jobs (default: -1 for all processors)
    """
    
    def __init__(
        self,
        num_kernels=5000,
        normalise=True,
        random_state=None,
        n_jobs=-1,
    ):
        """Initialize MultiROCKET regressor."""
        super().__init__(
            num_kernels=num_kernels,
            normalise=normalise,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.d_output = 3  # For 3D force: fx, fy, fz
    
    def fit(self, X, y):
        """Fit the model to training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_channels, n_timepoints)
            Training input sequences
        y : array-like of shape (n_samples, n_outputs)
            Target force values
        
        Returns
        -------
        self : object
            Returns self
        """
        # Reshape if needed: (batch, seq_len, features) -> (batch, features, seq_len)
        if X.ndim == 3 and X.shape[2] > X.shape[1]:
            X = np.transpose(X, (0, 2, 1))
        
        return super().fit(X, y)
    
    def predict(self, X):
        """Predict force values.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_channels, n_timepoints)
            Input sequences
        
        Returns
        -------
        y_pred : array-like of shape (n_samples, n_outputs)
            Predicted force values
        """
        # Reshape if needed: (batch, seq_len, features) -> (batch, features, seq_len)
        if X.ndim == 3 and X.shape[2] > X.shape[1]:
            X = np.transpose(X, (0, 2, 1))
        
        return super().predict(X)


class HydraForceRegressor(HydraRegressor):
    """Hydra-based force prediction model.
    
    Hydra is an ensemble method combining ROCKET with multiple
    Ridge regressors at different regularization strengths.
    
    Parameters
    ----------
    num_kernels : int, optional
        Number of random convolutional kernels (default: 2000)
    normalise : bool, optional
        Whether to normalize inputs (default: True)
    random_state : int, optional
        Random seed for reproducibility (default: None)
    n_jobs : int, optional
        Number of parallel jobs (default: -1 for all processors)
    """
    
    def __init__(
        self,
        num_kernels=2000,
        normalise=True,
        random_state=None,
        n_jobs=-1,
    ):
        """Initialize Hydra regressor."""
        super().__init__(
            num_kernels=num_kernels,
            normalise=normalise,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.d_output = 3  # For 3D force: fx, fy, fz
    
    def fit(self, X, y):
        """Fit the model to training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_channels, n_timepoints)
            Training input sequences
        y : array-like of shape (n_samples, n_outputs)
            Target force values
        
        Returns
        -------
        self : object
            Returns self
        """
        # Reshape if needed: (batch, seq_len, features) -> (batch, features, seq_len)
        if X.ndim == 3 and X.shape[2] > X.shape[1]:
            X = np.transpose(X, (0, 2, 1))
        
        return super().fit(X, y)
    
    def predict(self, X):
        """Predict force values.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_channels, n_timepoints)
            Input sequences
        
        Returns
        -------
        y_pred : array-like of shape (n_samples, n_outputs)
            Predicted force values
        """
        # Reshape if needed: (batch, seq_len, features) -> (batch, features, seq_len)
        if X.ndim == 3 and X.shape[2] > X.shape[1]:
            X = np.transpose(X, (0, 2, 1))
        
        return super().predict(X)
