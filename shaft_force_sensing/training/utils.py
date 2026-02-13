from argparse import ArgumentParser
from itertools import chain
from collections import defaultdict
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing import Tuple
import numpy as np
from datetime import datetime

from torch.utils.data import ConcatDataset, random_split

from shaft_force_sensing import ForceSensingDataset


def args_parser() -> dict:
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--model_type", type=str,
                        choices=["transformer", "ltc"], default="transformer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str,
                        default=f'logs/{datetime.now().strftime("%Y%m%d_%H%M%S")}')

    def parse_unknown_args(unknown):
        """Parse unknown arguments assuming --key value format with auto type conversion"""
        parsed = {}
        for i in range(0, len(unknown), 2):
            if i + 1 < len(unknown):
                key = unknown[i].lstrip('--')
                value = unknown[i + 1]

                # Auto type conversion
                try:
                    # Try boolean
                    if value.lower() in ['true', 'false']:
                        parsed[key] = value.lower() == 'true'
                    # Try int
                    elif '.' not in value:
                        parsed[key] = int(value)
                    # Try float
                    else:
                        parsed[key] = float(value)
                except ValueError:
                    # Keep as string if conversion fails
                    parsed[key] = value
        return parsed

    args, unknown = parser.parse_known_args()
    args = vars(args)
    args.update(parse_unknown_args(unknown))

    return args


def prepare_datasets(
    data_root: str,
    input_cols: list,
    target_cols: list
) -> Tuple[ForceSensingDataset, ForceSensingDataset, StandardScaler]:
    data_paths = sorted(Path(data_root).rglob("*.csv"))

    groups = defaultdict(list)
    for p in data_paths:
        groups[p.parent.name].append(p)

    test_paths = [lst[-1] for lst in groups.values()]
    train_paths = [p for p in data_paths if p not in test_paths]
    train_paths.pop(3)
    train_paths.pop(2)

    scaler = StandardScaler()
    forces = []
    for p in tqdm(train_paths):
        data = np.loadtxt(p, delimiter=",", skiprows=1)
        forces.append(data[:, -3:])
    forces = np.concatenate(forces, axis=0)
    scaler.fit(forces)

    train_sets = defaultdict(list)
    for p in tqdm(train_paths):
        stride = 5
        if p.parent.name == 'Free':
            stride *= 4
        dataset = ForceSensingDataset(
            p,
            input_cols,
            target_cols,
            stride,
            nomalizer=scaler)
        train_sets[p.parent.name].append(dataset)

    train_set = ConcatDataset(
        list(chain.from_iterable(train_sets.values())))

    train_size = int(0.9 * len(train_set))
    val_size = len(train_set) - train_size
    train_set, val_set = random_split(train_set, [train_size, val_size])

    return train_set, val_set, scaler
