import numpy as np
import plotly.graph_objs as go
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from itertools import chain
from datetime import datetime
from torch.utils.data import ConcatDataset, DataLoader, random_split

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

# Configuration
batch_size = 256
random_seed = 42

# Data configuration

# Input and target column definitions
i_cols = [
    'jaw_position', 'wrist_pitch_position', 'wrist_yaw_position', 'roll_position',
    'wrist_pitch_velocity', 'wrist_yaw_velocity', 'jaw_velocity', 'roll_velocity',
    'wrist_pitch_effort', 'wrist_yaw_effort', 'roll_effort',
    'jaw_effort', 'insertion_effort', 'yaw_effort', 'pitch_effort',
    'tx', 'ty', 'tz', 'fx', 'fy', 'fz'
]
t_cols = ['ati_fx', 'ati_fy', 'ati_fz']

data_paths = sorted(Path("/home/sxk2514/erie/shaft_force_sensing/data/").rglob("*.csv"))

groups = defaultdict(list)
for p in data_paths:
    groups[p.parent.name].append(p)

test_paths = [lst[-1] for lst in groups.values()]
train_paths = [p for p in data_paths if p not in test_paths]
train_paths.pop(3);
train_paths.pop(2);

golbal_scaler = StandardScaler()
forces = []
for p in tqdm(train_paths):
    data = np.loadtxt(p, delimiter=",", skiprows=1)
    forces.append(data[:, -3:])
forces = np.concatenate(forces, axis=0)
golbal_scaler.fit(forces);

train_sets = defaultdict(list)
for p in tqdm(train_paths):
    stride = 5
    if p.parent.name == 'Free':
        stride *= 4
    dataset = ForceSensingDataset(
        p, i_cols, t_cols,
        stride, nomalizer=golbal_scaler)
    train_sets[p.parent.name].append(dataset)

train_set = ConcatDataset(
    list(chain.from_iterable(train_sets.values())))

train_size = int(0.9 * len(train_set))
val_size = len(train_set) - train_size
train_set, val_set = random_split(train_set, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# Model configurations to compare
models_config = {
    # 'RocketRegressor': {
    #     'class': LitRocket,
    #     'kwargs': {
    #         'n_kernels': 15000,
    #         'n_jobs': -1   # ✅ add this
    #     },
    #     'description': 'ROCKET with 10K kernels'
    # },
    'MultiRocketRegressor': {
        'class': LitMultiRocket,
        'kwargs': {
            'n_kernels': 20000,
            'n_jobs': 1   # ✅ add here too
        },
        'description': 'Multi-scale ROCKET with 5K kernels'
    },
    # 'HydraRegressor': {
    #     'class': LitHydra,
    #     'kwargs': {
    #         'n_kernels': 64,
    #         'n_groups': 64,
    #         'n_jobs': -1   # ⚠️ only if Hydra supports it
    #     },
    #     'description': 'Hydra with 8 kernels x 64 groups'
    # },
}

results = {}

import time
import joblib
from pathlib import Path
from datetime import datetime

# Create experiment directory
save_root = Path("../logs")
save_dir = save_root / datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir.mkdir(parents=True, exist_ok=True)

print(f"Saving models to: {save_dir}")
results = {}
best_model = None
best_loss = float("inf")

for model_name, config in models_config.items():
    print(f"\n{'='*60}")
    print(f"Training {model_name}: {config['description']}")
    print(f"{'='*60}")
    
    # Initialize model
    model = config['class'](
        d_input=len(i_cols),
        d_output=len(t_cols),
        **config['kwargs']
    )
    
    # -------------------------
    # ✅ TRAIN (returns val_loss)
    # -------------------------
    start_time = time.time()
    val_loss = model.train_rocket(train_loader, val_loader)  # 👈 IMPORTANT
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    if val_loss is not None:
        print(f"Validation Loss: {val_loss:.6f}")
    
    # -------------------------
    # ✅ SAVE MODEL
    # -------------------------
    model_path = save_dir / f"{model_name}.pkl"
    joblib.dump(model, model_path)
    print(f"Saved model to: {model_path}")
    
    # -------------------------
    # ✅ TRACK BEST MODEL
    # -------------------------
    if val_loss is not None and val_loss < best_loss:
        best_loss = val_loss
        best_model = model
        joblib.dump(model, save_dir / "best_model.pkl")
        print(f"🔥 New best model saved (val_loss={val_loss:.6f})")

    # Store results
    results[model_name] = {
        'model': model,
        'training_time': training_time,
        'val_loss': val_loss,
        'config': config,
        'model_path': model_path
    }

print("\n✅ All models trained and saved successfully!")