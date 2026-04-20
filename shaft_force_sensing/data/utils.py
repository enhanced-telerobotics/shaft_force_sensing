from pathlib import Path
from typing import Tuple, List
import json5


def get_train_test(
        data_root: Path,
        model_cls: str = None,
        teleop: bool = False,
        model_idx: int = 0
) -> Tuple[List[Path], List[Path]]:
    if not teleop:
        data_path = data_root / "Automated"
    else:
        data_path = data_root / "Teleop"

    with open(data_path / "split.json", "r") as f:
        split = json5.load(f)[f"M{model_idx}"]

    all_csv_paths = list(data_path.rglob("*.csv"))

    train = split["train"]
    test = split["test"]

    train_paths = [p for p in all_csv_paths if p.stem in train]
    test_paths = [p for p in all_csv_paths if p.stem in test]

    return sorted(train_paths), sorted(test_paths)

def get_cols(ablations: str) -> Tuple[List[str], List[str]]:
    # Global column definitions
    i_cols = [
        'jaw_position', 'wrist_pitch_position', 'wrist_yaw_position',  'roll_position',
        'wrist_pitch_velocity', 'wrist_yaw_velocity', 'jaw_velocity', 'roll_velocity',
        'wrist_pitch_effort', 'wrist_yaw_effort', 'roll_effort',
        'jaw_effort', 'insertion_effort', 'yaw_effort', 'pitch_effort',
        'tx', 'ty', 'tz', 'fx', 'fy', 'fz'
    ]
    t_cols = ['ati_fx', 'ati_fy', 'ati_fz']

    # Set ablation columns based on specified ablation type
    if ablations is not None:
        if ablations == 'no_hex10':
            a_cols = ['tx', 'ty', 'tz', 'fx', 'fy', 'fz']
        elif ablations == 'no_tau':
            a_cols = ['*_effort']
        elif ablations == 'no_vec':
            a_cols = ['*_velocity']
        elif ablations == 'no_pos':
            a_cols = ['*_position']
        elif ablations == 'no_pos_vec':
            a_cols = ['*_position', '*_velocity']
        else:
            raise ValueError(f"Unsupported ablation type: {ablations}")
    else:
        a_cols = []

    # Remove ablated columns from input column lists
    i_cols = [col for col in i_cols if not any(
        Path(col).match(a) for a in a_cols)]

    return i_cols, t_cols


if __name__ == "__main__":
    data_root = Path("data")
    model_type = "LSTM"
    model_idx = 0
    teleop = False

    train_paths, test_paths = get_train_test(
        data_root, model_type, teleop, model_idx)

    print("Train paths:")
    for p in train_paths:
        print(p)
    print("\nTest paths:")
    for p in test_paths:
        print(p)

    ablations = input(
        "Enter ablation type (e.g., 'no_hex10', 'no_tau', 'no_vec', 'no_pos', 'no_pos_vec', or None): ")
    i_cols, _ = get_cols(ablations)
    print(f"\nInput columns after ablation ({ablations}): {i_cols}")
    print(f"Total input columns after ablation: {len(i_cols)}")
