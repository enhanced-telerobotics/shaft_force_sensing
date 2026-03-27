"""ROCKET-based models for force prediction using sktime.

All three classes are thin configuration wrappers around sktime's built-in
RocketRegressor, which internally combines the chosen ROCKET transformer
with a Ridge regressor.

References:
    - RocketRegressor: https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.regression.kernel_based.RocketRegressor.html

Notes:
    - Hydra is not available in sktime. HydraForceRegressor uses
      rocket_transform="multirocket" with use_multivariate="yes" as the
      closest available substitute. For native Hydra, use aeon's
      HydraTransformer or https://github.com/angus924/hydra
"""

import os
import numpy as np
from sktime.regression.kernel_based import RocketRegressor
from sklearn.linear_model import Ridge


class RocketForceRegressor:
    """ROCKET-based force prediction model.

    Parameters
    ----------
    num_kernels : int, optional
        Number of random convolutional kernels (default: 10000).
    random_state : int, optional
        Random seed for reproducibility (default: None).
    n_jobs : int, optional
        Number of parallel jobs (default: 1).
    """

    def __init__(self, num_kernels=10000, random_state=None, n_jobs=1):
        self.model = RocketRegressor(
            num_kernels=num_kernels,
            rocket_transform="rocket",
            use_multivariate="auto",
            random_state=random_state,
            n_jobs=self._safe_n_jobs(n_jobs),
        )
        # self.model.estimator = Ridge(alpha=1.0, solver="sparse_cg")

    def fit(self, X, y):
        self.model.fit(self._validate_X(X).astype(np.float64), y)
        return self

    def predict(self, X):
        return self.model.predict(self._validate_X(X).astype(np.float64))

    @staticmethod
    def _validate_X(X):
        if X.ndim == 3 and X.shape[2] > X.shape[1]:
            X = np.transpose(X, (0, 2, 1))
        return X

    @staticmethod
    def _safe_n_jobs(n_jobs):
        """Cap n_jobs to available CPU count to avoid numba thread overflow."""
        max_jobs = os.cpu_count() or 1
        if n_jobs == -1 or n_jobs > max_jobs:
            return max_jobs
        return n_jobs


class MultiRocketForceRegressor:
    """Multi-Scale ROCKET-based force prediction model.

    Parameters
    ----------
    num_kernels : int, optional
        Number of random convolutional kernels (default: 6250).
        Rounded down internally to the nearest multiple of 84.
    random_state : int, optional
        Random seed for reproducibility (default: None).
    n_jobs : int, optional
        Number of parallel jobs (default: 1).
    """

    def __init__(self, num_kernels=6250, random_state=None, n_jobs=1):
        self.model = RocketRegressor(
            num_kernels=num_kernels,
            rocket_transform="multirocket",
            use_multivariate="yes",
            random_state=random_state,
            n_jobs=self._safe_n_jobs(n_jobs),
        )
        self.model.estimator = Ridge(alpha=1.0, solver="sparse_cg")

    def fit(self, X, y):
        self.model.fit(self._validate_X(X).astype(np.float64), y)
        return self

    def predict(self, X):
        return self.model.predict(self._validate_X(X).astype(np.float64))

    @staticmethod
    def _validate_X(X):
        if X.ndim == 3 and X.shape[2] > X.shape[1]:
            X = np.transpose(X, (0, 2, 1))
        return X

    @staticmethod
    def _safe_n_jobs(n_jobs):
        """Cap n_jobs to available CPU count to avoid numba thread overflow."""
        max_jobs = os.cpu_count() or 1
        if n_jobs == -1 or n_jobs > max_jobs:
            return max_jobs
        return n_jobs


class HydraForceRegressor:
    """Hydra-style force prediction using MultiRocket (sktime substitute).

    NOTE: sktime has no native Hydra transformer. This class uses
    rocket_transform="multirocket" with use_multivariate="yes" as the
    closest sktime substitute. For true Hydra, use aeon's HydraTransformer
    or the standalone package: https://github.com/angus924/hydra

    Parameters
    ----------
    num_kernels : int, optional
        Number of kernels (default: 6250).
    random_state : int, optional
        Random seed for reproducibility (default: None).
    n_jobs : int, optional
        Number of parallel jobs (default: 1).
    """

    def __init__(self, num_kernels=6250, random_state=None, n_jobs=1):
        self.model = RocketRegressor(
            num_kernels=num_kernels,
            rocket_transform="multirocket",
            use_multivariate="yes",
            random_state=random_state,
            n_jobs=self._safe_n_jobs(n_jobs),
        )
        self.model.estimator = Ridge(alpha=1.0, solver="sparse_cg")

    def fit(self, X, y):
        self.model.fit(self._validate_X(X).astype(np.float64), y)
        return self

    def predict(self, X):
        return self.model.predict(self._validate_X(X).astype(np.float64))

    @staticmethod
    def _validate_X(X):
        if X.ndim == 3 and X.shape[2] > X.shape[1]:
            X = np.transpose(X, (0, 2, 1))
        return X

    @staticmethod
    def _safe_n_jobs(n_jobs):
        """Cap n_jobs to available CPU count to avoid numba thread overflow."""
        max_jobs = os.cpu_count() or 1
        if n_jobs == -1 or n_jobs > max_jobs:
            return max_jobs
        return n_jobs