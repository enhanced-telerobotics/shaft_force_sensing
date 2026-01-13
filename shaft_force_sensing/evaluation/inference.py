"""Model inference and testing."""

import numpy as np
import torch
from scipy.signal import medfilt


def smooth_force_predictions(force_preds, kernel_size=5):
    """Apply median filter to force predictions.
    
    Parameters
    ----------
    force_preds : np.ndarray
        Force predictions of shape [N, 3]
    kernel_size : int, optional
        Size of median filter window (default: 5)
    
    Returns
    -------
    np.ndarray
        Smoothed force predictions
    """
    return medfilt(force_preds, kernel_size=(kernel_size, 1))


def test_model(active_loader, model):
    """Test model on a data loader and return predictions.
    
    Parameters
    ----------
    active_loader : torch.utils.data.DataLoader
        Data loader with test/validation data
    model : LitTransformer
        Model to test
    
    Returns
    -------
    tuple
        (smoothed_predictions, targets) both of shape [N, 3]
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_force_preds = []
    all_force_targets = []

    with torch.no_grad():
        for x_batch, force_y_batch in active_loader:
            x_batch = x_batch.to(device)
            force_y_batch = force_y_batch.to(device)

            force_pred = model(x_batch)

            all_force_preds.append(force_pred.cpu().numpy())
            all_force_targets.append(force_y_batch.cpu().numpy())

    # Concatenate all batches
    all_force_preds = np.concatenate(all_force_preds, axis=0)
    all_force_targets = np.concatenate(all_force_targets, axis=0)

    # Unnormalize using dataset stats
    force_mean = active_loader.dataset.force_mean
    force_std = active_loader.dataset.force_std

    if isinstance(force_mean, torch.Tensor):
        force_mean = force_mean.cpu().numpy()
    if isinstance(force_std, torch.Tensor):
        force_std = force_std.cpu().numpy()

    all_force_preds = all_force_preds * force_std + force_mean
    all_force_targets = all_force_targets * force_std + force_mean

    # Smooth predictions
    smoothed_force_preds = smooth_force_predictions(all_force_preds, kernel_size=71)

    return smoothed_force_preds, all_force_targets
