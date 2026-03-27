import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from itertools import chain
from datetime import datetime
from torch.utils.data import ConcatDataset, DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
import joblib

from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    mean_absolute_error,
)
import time

from shaft_force_sensing import ForceSensingDataset
from shaft_force_sensing.models import (
    LitRocket,
    LitMultiRocket,
    LitHydra,
)
from shaft_force_sensing.evaluation import (
    tb_to_numpy,
    add_norm,
    array_bais,
    array_medfilt,
)

# ── Configuration ────────────────────────────────────────────────────────────
batch_size  = 256
random_seed = 42
dummy_run   = False      # ← set False for full training

# ── Column definitions ───────────────────────────────────────────────────────
i_cols = [
    'jaw_position', 'wrist_pitch_position', 'wrist_yaw_position', 'roll_position',
    'wrist_pitch_velocity', 'wrist_yaw_velocity', 'jaw_velocity', 'roll_velocity',
    'wrist_pitch_effort', 'wrist_yaw_effort', 'roll_effort',
    'jaw_effort', 'insertion_effort', 'yaw_effort', 'pitch_effort',
    'tx', 'ty', 'tz', 'fx', 'fy', 'fz'
]
t_cols = ['ati_fx', 'ati_fy', 'ati_fz']

# ── Data loading ─────────────────────────────────────────────────────────────
data_paths = sorted(Path("/home/sxk2514/erie/shaft_force_sensing/data/").rglob("*.csv"))

groups = defaultdict(list)
for p in data_paths:
    groups[p.parent.name].append(p)

test_paths  = [lst[-1] for lst in groups.values()]
train_paths = [p for p in data_paths if p not in test_paths]
train_paths.pop(3)
train_paths.pop(2)

# ── Fit global scaler on training forces ─────────────────────────────────────
golbal_scaler = StandardScaler()
forces = []
for p in tqdm(train_paths, desc="Fitting scaler"):
    data = np.loadtxt(p, delimiter=",", skiprows=1)
    forces.append(data[:, -3:])
forces = np.concatenate(forces, axis=0)
golbal_scaler.fit(forces)

# ── Build datasets ───────────────────────────────────────────────────────────
train_sets = defaultdict(list)
for p in tqdm(train_paths, desc="Building datasets"):
    stride = 5
    if p.parent.name == 'Free':
        stride *= 4
    dataset = ForceSensingDataset(
        p, i_cols, t_cols,
        stride, nomalizer=golbal_scaler)
    train_sets[p.parent.name].append(dataset)

train_set = ConcatDataset(list(chain.from_iterable(train_sets.values())))

train_size = int(0.9 * len(train_set))
val_size   = len(train_set) - train_size
train_set, val_set = random_split(train_set, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)

# ── Dummy run subset ─────────────────────────────────────────────────────────
if dummy_run:
    n = batch_size * 2
    train_loader = DataLoader(Subset(train_set, range(n)), batch_size=batch_size)
    val_loader   = DataLoader(Subset(val_set,   range(n)), batch_size=batch_size)
    print(f"[dummy_run] Using {n} train / {n} val samples")

# ── Save directory ───────────────────────────────────────────────────────────
save_root = Path("../logs_ROCKET")
save_dir  = save_root / datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir.mkdir(parents=True, exist_ok=True)

# ── Model ────────────────────────────────────────────────────────────────────
model = LitMultiRocket(
    num_kernels=500,
    random_state=random_seed,
    n_jobs=85,
)
model_cls = model._get_name()
print(f"Model: {model_cls}")

# ── Train ────────────────────────────────────────────────────────────────────
writer = SummaryWriter(log_dir=str(save_dir / model_cls))

t0 = time.time()
model.train_rocket(train_loader, val_loader, writer=writer)
print(f"Training time: {time.time() - t0:.2f}s")

writer.close()

# ── Save model ───────────────────────────────────────────────────────────────
model_path = save_dir / f"{model_cls}.pkl"
joblib.dump(model.rocket_model, model_path)
print(f"Model saved to {model_path}")