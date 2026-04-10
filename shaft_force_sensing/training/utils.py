from collections import defaultdict
from argparse import ArgumentParser
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, random_split, Dataset
from tqdm import tqdm

import shaft_force_sensing.models
from shaft_force_sensing.models import LitSequenceModel
from shaft_force_sensing.data import (
    SensorDataset,
    TorqueDataset,
    get_train_test,
    get_cols
)


def args_parser() -> dict:
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--model_type", type=str,
                        choices=[
                            "transformer",
                            "ltc",
                            "lstm"], default="transformer")
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str,
                        default=f'logs/{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    parser.add_argument("--model_dir", type=str, default=None)

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
    model_cls: str,
    finetune: bool = False,
    ablations: str = None,
    model_idx: int = 0,
    stride: int = 5,
    sequence_length: int = 100
) -> Tuple[Dataset, Dataset, StandardScaler]:
    # Get train/test splits based on model type and finetuning setting
    train_paths, _ = get_train_test(
        data_root,
        model_cls,
        finetune,
        model_idx
    )

    train_sets = defaultdict(list)

    if 'lstm' not in model_cls.lower():
        # Initialize normalizer using all training data
        scaler = StandardScaler()
        forces = []
        for p in tqdm(train_paths):
            data = np.loadtxt(p, delimiter=",", skiprows=1)
            forces.append(data[:, -3:])
        forces = np.concatenate(forces, axis=0)
        scaler.fit(forces)

        # Get input and target columns based on ablation settings
        input_cols, target_cols = get_cols(ablations)

        # Downsample more for free space data
        for p in tqdm(train_paths):
            dataset = SensorDataset(
                p,
                input_cols,
                target_cols,
                # Use larger stride for free space data to balance the dataset
                stride if p.parent.name != 'Free' else stride * 4,
                sequence_length,
                nomalizer=scaler)
            train_sets[p.parent.name].append(dataset)
    else:
        scaler = None

        for path in tqdm(train_paths):
            dataset = TorqueDataset(
                path,
                stride,
                sequence_length,
            )
            train_sets[path.parent.name].append(dataset)
    
    # Concatenate datasets from different groups and split into train/val
    train_set = ConcatDataset(
        list(chain.from_iterable(train_sets.values())))

    train_size = int(0.9 * len(train_set))
    val_size = len(train_set) - train_size
    train_set, val_set = random_split(train_set, [train_size, val_size])

    return train_set, val_set, scaler


def prepare_test_dataset(
    data_root: str,
    model: LitSequenceModel,
    finetune: bool = None,
    ablations: str = None
) -> Tuple[dict, int]:
    # Initialize test sets dictionary
    test_sets = dict()
    
    # Retrieve settings from the model
    if finetune is None:
        finetune = model.hparams.get("finetune", False)
    if ablations is None:
        ablations = model.hparams.get("ablations", None)
    model_cls = model._get_name()

    # Get test paths
    _, test_paths = get_train_test(
        data_root,
        model_cls,
        finetune,
        model.hparams.get("model_idx", 0)
    )

    # Prepare test datasets
    if model_cls == "LitTransformer" or model_cls == "LitLTC":
        golbal_scaler = StandardScaler()
        golbal_scaler.mean_ = model.data_mean.numpy(force=True)
        golbal_scaler.scale_ = model.data_std.numpy(force=True)

        i_cols, t_cols = get_cols(ablations)

        for p in tqdm(test_paths):
            dataset = SensorDataset(
                p, i_cols, t_cols,
                nomalizer=golbal_scaler)
            test_sets[p.stem] = dataset

        # As large as possible for faster inference
        batch_size = 1000
    elif model_cls == "LitLSTM":
        for p in tqdm(test_paths):
            dataset = TorqueDataset(p, stride=1)
            test_sets[p.stem] = dataset

        # LSTM uses hidden states
        batch_size = 1
    else:
        raise ValueError(f"Unknown model class: {model_cls}")
    
    return test_sets, batch_size


def load_model(dir: Path, **override_hparams) -> LitSequenceModel:
    model_cls_name = None
    for p in dir.iterdir():
        if p.name in shaft_force_sensing.models.__all__:
            model_cls_name = p.name
            break

    assert model_cls_name is not None, "Model name not found in checkpoint directory."

    ckpt_paths = sorted(dir.glob("best*.ckpt"))
    assert ckpt_paths, "No checkpoint matching 'best*.ckpt' found in model directory."

    model_cls = getattr(shaft_force_sensing.models, model_cls_name)
    return model_cls.load_from_checkpoint(ckpt_paths[-1], **override_hparams)


if __name__ == "__main__":
    # Example usage
    train_set, val_set, scaler = prepare_datasets(
        Path().cwd() / "data",
        'lstm',
        finetune=False,
        ablations=None,
        model_idx=0,
        stride=1,
        sequence_length=100,
    )

    print(f"Train set size: {len(train_set)}")
    print(f"First train sample shape: {train_set[0][0].shape}, {train_set[0][1].shape}")