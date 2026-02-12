from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _load_scalars_from_event_dir(event_dir: Path) -> Dict[str, np.ndarray]:
    """
    Load all scalar tags from a tensorboard event directory.
    Returns:
        dict[tag] -> np.ndarray of shape (T,)
    """
    ea = EventAccumulator(
        str(event_dir),
        size_guidance={"scalars": 0},
    )
    ea.Reload()

    scalar_dict = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        values = np.array([e.value for e in events], dtype=np.float32)
        scalar_dict[tag] = values

    return scalar_dict


def tb_to_numpy(log_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load tensorboard logs from a directory with structure:

        log_dir/
            test_force_0_gt/
            test_force_0_pred/
            test_force_1_gt/
            test_force_1_pred/
            ...

    Returns:
        gt  : ndarray (N, D)
        pred: ndarray (N, D)

    Where:
        N = total time steps (concatenated)
        D = number of force dimensions (e.g. 3)
    """

    log_dir = Path(log_dir)

    gt_dirs = sorted(log_dir.glob("*_gt"))
    pred_dirs = sorted(log_dir.glob("*_pred"))

    if len(gt_dirs) == 0 or len(pred_dirs) == 0:
        raise ValueError(f"No *_gt or *_pred directories found in {log_dir}")

    if len(gt_dirs) != len(pred_dirs):
        raise ValueError("Mismatch between number of gt and pred folders")

    gt_list: List[np.ndarray] = []
    pred_list: List[np.ndarray] = []

    for gt_dir, pred_dir in zip(gt_dirs, pred_dirs):

        gt_scalars = _load_scalars_from_event_dir(gt_dir)
        pred_scalars = _load_scalars_from_event_dir(pred_dir)

        # assume one scalar tag per folder
        if len(gt_scalars) != 1 or len(pred_scalars) != 1:
            raise ValueError(
                f"Expected exactly 1 scalar tag in {gt_dir} and {pred_dir}"
            )

        gt_values = next(iter(gt_scalars.values()))
        pred_values = next(iter(pred_scalars.values()))

        if len(gt_values) != len(pred_values):
            raise ValueError(
                f"Length mismatch in {gt_dir.name}: "
                f"{len(gt_values)} vs {len(pred_values)}"
            )

        gt_list.append(gt_values)
        pred_list.append(pred_values)

    # Stack along feature dimension (force dimension)
    # Result: (T, D)
    gt = np.stack(gt_list, axis=1)
    pred = np.stack(pred_list, axis=1)

    return gt, pred
