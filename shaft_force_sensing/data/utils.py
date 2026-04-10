from pathlib import Path
from typing import Tuple, List
import json5


def get_train_test(
        data_root: Path,
        model_type: str = None,
        model_idx: int = 0,
        transfer_learning: bool = False
) -> Tuple[List[Path], List[Path]]:
    if not transfer_learning:
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

    if 'lstm' in model_type.lower():
        train_paths = [
            p for p in train_paths if "F" in p.stem or "free" in p.stem]

    return sorted(train_paths), sorted(test_paths)


if __name__ == "__main__":
    data_root = Path("data")
    model_type = "LSTM"
    model_idx = 0
    transfer_learning = False

    train_paths, test_paths = get_train_test(
        data_root, model_type, model_idx, transfer_learning)

    print("Train paths:")
    for p in train_paths:
        print(p)
    print("\nTest paths:")
    for p in test_paths:
        print(p)
