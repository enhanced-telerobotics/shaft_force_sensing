import numpy as np
from scipy.signal import medfilt


def add_norm(arr: np.ndarray) -> np.ndarray:
    """Add a column of L2 norms to the input array."""
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    return np.concatenate([arr, norm], axis=1)


def array_bais(arr: np.ndarray, length: int = 200) -> np.ndarray:
    """Bias the input array by minus the mean of the first `length` elements."""
    bias = np.mean(arr[:length], axis=0, keepdims=True)
    return arr - bias

def array_medfilt(arr: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply median filter to each column of the input array."""
    return medfilt(arr, kernel_size=(kernel_size, 1))