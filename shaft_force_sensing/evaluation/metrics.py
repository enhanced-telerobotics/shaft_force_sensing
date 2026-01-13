"""Evaluation metrics and result saving."""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def evaluate_results(test_results, test_targets, mode, run_id):
    """Evaluate model results against targets.
    
    Parameters
    ----------
    test_results : np.ndarray
        Model predictions of shape [N, 3]
    test_targets : np.ndarray
        Ground truth targets of shape [N, 3]
    mode : str
        Name of the ablation mode or model configuration
    run_id : int
        Run ID for tracking multiple runs
    
    Returns
    -------
    pd.DataFrame
        DataFrame with metrics for each axis
    """
    eps = 1e-8
    axis_labels = ["Fx", "Fy", "Fz"]
    results = []

    for i in range(3):
        pred_i = test_results[:, i]
        true_i = test_targets[:, i]
        axis = axis_labels[i]

        signed_err = true_i - pred_i
        rmse = np.sqrt(np.mean(signed_err**2))
        r2 = r2_score(true_i, pred_i)

        max_force = np.max(true_i)
        min_force = np.min(true_i)
        force_range = max_force - min_force

        nrmsd = (rmse / (force_range + eps)) * 100

        results.append(
            {
                "Ablation_Mode": mode,
                "Run_ID": run_id,
                "Axis": axis,
                "Min Force (N)": min_force,
                "Max Force (N)": max_force,
                "Range (N)": force_range,
                "RMSE (N)": rmse,
                "NRMSD (%)": nrmsd,
                "R2": r2,
            }
        )

    return pd.DataFrame(results).round(3)


def save_force_results(smoothed_force_preds, all_force_targets, label, save_dir="results"):
    """Save force predictions and targets to CSV.
    
    Parameters
    ----------
    smoothed_force_preds : np.ndarray
        Smoothed force predictions of shape [N, 3]
    all_force_targets : np.ndarray
        Target forces of shape [N, 3]
    label : str
        Label for the output file name
    save_dir : str, optional
        Directory to save results (default: "results")
    """
    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame(
        np.hstack([smoothed_force_preds, all_force_targets]),
        columns=[
            "Fx_pred",
            "Fy_pred",
            "Fz_pred",
            "Fx_target",
            "Fy_target",
            "Fz_target",
        ],
    )

    csv_path = os.path.join(save_dir, f"{label}_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved results to {csv_path}")
