"""Batch evaluation CLI for checkpoint directories.

Example:
    python -m shaft_force_sensing.evaluation.batch_eval logs/transformer

This will evaluate each child directory under ``logs/transformer`` such as
``base``, ``ft_m0``, ``ft_m1``, ... and write metrics to:

    <run_dir>/Automated/metrics.txt
    or
    <run_dir>/Teleop/metrics.txt
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from tqdm import tqdm

from shaft_force_sensing.evaluation.logger import tb_to_numpy
from shaft_force_sensing.evaluation.post_processing import add_norm, array_bais, array_medfilt
from shaft_force_sensing.training.utils import load_model

AXES = ["F_x", "F_y", "F_z", "Norm"]


def _rmse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((gt - pred) ** 2, axis=0))


def _r2(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    ss_res = np.sum((gt - pred) ** 2, axis=0)
    ss_tot = np.sum((gt - np.mean(gt, axis=0, keepdims=True)) ** 2, axis=0)
    return 1.0 - (ss_res / (ss_tot + eps))


def _iter_group_dirs(test_path: Path) -> Iterable[Path]:
    return [p for p in sorted(test_path.iterdir()) if p.is_dir()]


def evaluate_run(run_dir: Path, force: bool = False) -> Tuple[Path, int]:
    """Evaluate a single run directory and write metrics.txt.

    Returns:
        metrics_path, number_of_groups_evaluated
    """
    run_dir = Path(run_dir)
    model = load_model(run_dir, map_location="cpu")

    teleop = model.hparams.get("teleop", False)
    model_cls = model._get_name()

    test_path = run_dir / ("Teleop" if teleop else "Automated")
    if not test_path.exists():
        raise FileNotFoundError(f"Test log folder not found: {test_path}")

    metrics_path = test_path / "metrics.txt"
    if metrics_path.exists() and not force:
        return metrics_path, 0

    data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for group_path in _iter_group_dirs(test_path):
        group = group_path.stem

        gt, pred = tb_to_numpy(group_path)

        # Apply median filter to reduce noise
        pred = array_medfilt(pred)

        # Only apply bias correction for non-LSTM teleop models
        if "lstm" not in model_cls.lower() and teleop:
            pred = array_bais(pred)

        gt = add_norm(gt)
        pred = add_norm(pred)
        data[group] = (gt, pred)

    if not data:
        raise ValueError(f"No valid group directories found in: {test_path}")

    gt_all = np.concatenate([data[group][0] for group in data], axis=0)
    pred_all = np.concatenate([data[group][1] for group in data], axis=0)
    data["All"] = (gt_all, pred_all)

    with open(metrics_path, "w", encoding="utf-8") as f:
        print(f"Model: {model_cls}", file=f)
        print(f"Test data: {test_path.stem}", file=f)
        print("-" * 10, file=f)
        for key, value in model.hparams.items():
            print(f"{key}: {value}", file=f)
        print("-" * 10, file=f)

    for group, (gt, pred) in data.items():
        eps = 1e-8
        gt_min = np.min(gt, axis=0)
        gt_max = np.max(gt, axis=0)
        gt_range = gt_max - gt_min + eps
        rmse = _rmse(gt, pred)
        nrmse = rmse / gt_range
        r2_scores = _r2(gt, pred, eps=eps)

        with open(metrics_path, "a", encoding="utf-8") as f:
            print(f"Group: {group}", file=f)
            for i, name in enumerate(AXES):
                print(
                    f"{name}: "
                    f"Range={gt_range[i]:.4f}, "
                    f"RMSE={rmse[i]:.4f}, "
                    f"NRMSE={nrmse[i] * 100:.2f}%, "
                    f"R2={r2_scores[i] * 100:.2f}",
                    file=f,
                )
            print("-" * 10, file=f)

    return metrics_path, len(data) - 1


def run_batch(root_dir: Path, force: bool = False, strict: bool = False) -> None:
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {root_dir}")

    run_dirs = [p for p in sorted(root_dir.iterdir()) if p.is_dir()]
    if not run_dirs:
        raise ValueError(f"No subdirectories found in: {root_dir}")

    print(f"Found {len(run_dirs)} runs under {root_dir}")

    failures = []
    for run_dir in tqdm(run_dirs, desc="Evaluating runs"):
        try:
            metrics_path, group_count = evaluate_run(run_dir, force=force)
            if group_count == 0 and not force:
                print(f"[skip] {run_dir.name}: metrics already exists -> {metrics_path}")
            else:
                print(f"[ok]   {run_dir.name}: {group_count} groups -> {metrics_path}")
        except Exception as exc:  # noqa: BLE001 - CLI should keep going unless strict
            msg = f"[fail] {run_dir.name}: {exc}"
            print(msg)
            failures.append(msg)
            if strict:
                raise

    if failures:
        print("\nCompleted with failures:")
        for msg in failures:
            print(msg)
    else:
        print("\nCompleted successfully.")


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Batch evaluate all run folders under a parent directory.")
    parser.add_argument(
        "root_dir",
        type=Path,
        help="Parent directory containing run folders (e.g. logs/transformer).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing metrics.txt files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop immediately on first failed run.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_batch(args.root_dir, force=args.force, strict=args.strict)


if __name__ == "__main__":
    main()
