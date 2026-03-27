import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
import joblib
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for HPC (no display needed)
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    root_mean_squared_error,
    r2_score,
)

import shaft_force_sensing.models
from shaft_force_sensing import ForceSensingDataset
from shaft_force_sensing.models import (
    LitRocket,
    LitMultiRocket,
    LitHydra,
)
from shaft_force_sensing.evaluation import (
    add_norm,
    array_bais,
    array_medfilt,
)

# ── Column definitions ───────────────────────────────────────────────────────
i_cols = [
    'jaw_position', 'wrist_pitch_position', 'wrist_yaw_position', 'roll_position',
    'wrist_pitch_velocity', 'wrist_yaw_velocity', 'jaw_velocity', 'roll_velocity',
    'wrist_pitch_effort', 'wrist_yaw_effort', 'roll_effort',
    'jaw_effort', 'insertion_effort', 'yaw_effort', 'pitch_effort',
    'tx', 'ty', 'tz', 'fx', 'fy', 'fz'
]
t_cols = ['ati_fx', 'ati_fy', 'ati_fz']
axes   = ['F_x', 'F_y', 'F_z', 'Norm']

data_paths = sorted(Path("/home/sxk2514/erie/shaft_force_sensing/data/").rglob("*.csv"))

groups = defaultdict(list)
for p in data_paths:
    groups[p.parent.name].append(p)

test_paths  = [lst[-1] for lst in groups.values()]
train_paths = [p for p in data_paths if p not in test_paths]
train_paths.pop(3)
train_paths.pop(2)

# ── Locate saved model ───────────────────────────────────────────────────────
save_path = Path("../logs_ROCKET/20260327_101413")

model_cls = None
for p in save_path.iterdir():
    if p.name in shaft_force_sensing.models.__all__:
        model_cls = p.name
        break
assert model_cls is not None, "Model name not found in checkpoint directory."

# ── Load model from .pkl ─────────────────────────────────────────────────────
model = eval(model_cls)()
model.rocket_model = joblib.load(save_path / f"{model_cls}.pkl")
model._is_fitted   = True
print(f"Loaded {model_cls} from {save_path / f'{model_cls}.pkl'}")

import os
# Re-cap n_jobs after loading in case it was saved with a higher value
max_jobs = os.cpu_count() or 2
if hasattr(model.rocket_model.model, 'n_jobs'):
    model.rocket_model.model.n_jobs = min(model.rocket_model.model.n_jobs, max_jobs)


# ── Rebuild scaler from training data ────────────────────────────────────────
golbal_scaler = StandardScaler()
forces = []
for p in tqdm(train_paths, desc="Fitting scaler"):
    data = np.loadtxt(p, delimiter=",", skiprows=1)
    forces.append(data[:, -3:])
golbal_scaler.fit(np.concatenate(forces, axis=0))

# ── Build test loaders ───────────────────────────────────────────────────────
test_sets = {}
for p in tqdm(test_paths, desc="Building test sets"):
    dataset = ForceSensingDataset(p, i_cols, t_cols, nomalizer=golbal_scaler)
    test_sets[p.parent.name] = dataset

test_loaders = {
    group: DataLoader(dset, batch_size=1000, shuffle=False)
    for group, dset in test_sets.items()
}

# ── Run inference per group ──────────────────────────────────────────────────
data = {}

for group, loader in tqdm(test_loaders.items(), desc="Evaluating"):
    X_list, y_list = [], []
    for x, y, _ in loader:
        X_list.append(x.detach().cpu().numpy())
        y_list.append(y.detach().cpu().numpy())

    X = np.vstack(X_list).astype(np.float64)
    y = np.vstack(y_list).astype(np.float64)

    y_pred = model.rocket_model.predict(X)

    y_pred = np.atleast_2d(y_pred)
    if y_pred.shape[0] != X.shape[0]:
        y_pred = y_pred.T

    gt   = golbal_scaler.inverse_transform(y)
    pred = golbal_scaler.inverse_transform(y_pred)

    pred = array_medfilt(pred, kernel_size=71)
    pred = array_bais(pred, 50)

    gt   = add_norm(gt)
    pred = add_norm(pred)

    data[group] = (gt, pred)

    # ── Plot per group ───────────────────────────────────────────────────
    d = gt.shape[1]
    fig, axes_list = plt.subplots(d, 1, figsize=(12, 4 * d), sharex=True)
    if d == 1:
        axes_list = [axes_list]

    for i, name in enumerate(axes):
        axes_list[i].plot(gt[:, i],   label=f"{name} (gt)",   linewidth=0.8)
        axes_list[i].plot(pred[:, i], label=f"{name} (pred)", linewidth=0.8)
        axes_list[i].set_title(name)
        axes_list[i].set_ylabel("Force (N)")
        axes_list[i].legend()

    axes_list[-1].set_xlabel("Sample")
    fig.suptitle(f"Ground Truth vs Prediction — {group}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path / f"plot_{group}.png", dpi=150)
    plt.close(fig)
    print(f"Saved plot for group: {group}")

# ── Concatenate all groups ───────────────────────────────────────────────────
gt_all   = np.concatenate([data[g][0] for g in data], axis=0)
pred_all = np.concatenate([data[g][1] for g in data], axis=0)
data['All'] = (gt_all, pred_all)

# ── Compute and log metrics ──────────────────────────────────────────────────
metrics_path = save_path / "metrics.txt"

for group, (gt, pred) in data.items():
    gt_range  = np.max(gt, axis=0) - np.min(gt, axis=0)
    rmse      = root_mean_squared_error(gt, pred, multioutput='raw_values')
    nrmse     = rmse / gt_range
    r2_scores = r2_score(gt, pred, multioutput='raw_values')

    with open(metrics_path, "a") as f:
        print(f"Group: {group}", file=f)
        for i, name in enumerate(axes):
            print(
                f"  {name}: "
                f"Range={gt_range[i]:.4f}, "
                f"RMSE={rmse[i]:.4f}, "
                f"NRMSE={nrmse[i]*100:.2f}%, "
                f"R2={r2_scores[i]*100:.2f}",
                file=f
            )
        print("-" * 10, file=f)

    # Also print to stdout
    print(f"\nGroup: {group}")
    for i, name in enumerate(axes):
        print(
            f"  {name}: "
            f"Range={gt_range[i]:.4f}, "
            f"RMSE={rmse[i]:.4f}, "
            f"NRMSE={nrmse[i]*100:.2f}%, "
            f"R2={r2_scores[i]*100:.2f}"
        )
    print("-" * 10)

print(f"\nMetrics saved to {metrics_path}")