"""Data loader assembly utilities."""

import numpy as np
from torch.utils.data import DataLoader

from .dataset import WindowDataset


def assemble_loaders(input_cols, target_cols, batch_size=256):
    """Assemble data loaders for training, validation, and testing.
    
    This function loads and preprocesses data from the global dataset variables
    (free1, free2, val2, pal1-6, val3, ten1-5, val1) and creates PyTorch data loaders.
    
    Parameters
    ----------
    input_cols : list
        List of input column names to use
    target_cols : list
        List of target column names to use
    batch_size : int, optional
        Batch size for data loaders (default: 256)
    
    Returns
    -------
    tuple
        (train_loader, val_loader, test_loader, f_mean, f_std)
    """
    # Import the loaded data (must be loaded before calling this function)
    # This assumes the datasets are available in the global scope
    # Consider refactoring to pass datasets explicitly
    import sys

    caller_globals = sys._getframe(1).f_globals

    # Get datasets from caller's globals
    free_train = [caller_globals["free1"], caller_globals["free2"]]
    free_test = [caller_globals["val2"]]
    palp_train = [
        caller_globals["pal1"],
        caller_globals["pal2"],
        caller_globals["pal3"],
        caller_globals["pal4"],
        caller_globals["pal5"],
        caller_globals["pal6"],
    ]
    palp_test = [caller_globals["val3"]]
    tens_train = [
        caller_globals["ten1"],
        caller_globals["ten2"],
        caller_globals["ten3"],
        caller_globals["ten4"],
        caller_globals["ten5"],
    ]
    tens_test = [caller_globals["val1"]]

    # Helper: concatenate from lists
    def concat_from_list(datasets, col):
        return np.concatenate([d[col] for d in datasets], axis=0)

    def cut_test_split(train_list, test_list, test_ratio=0.1):
        """Cut test set to match ratio of training data."""
        total_train_len = sum(len(ds) for ds in train_list)
        target_test_len = int(np.ceil(total_train_len * test_ratio))
        target_test_len = min(target_test_len, len(test_list[0]))
        print(f"train total len {total_train_len}, target test len {target_test_len}")
        return [test_list[0][:target_test_len]]

    free_test_cut = cut_test_split(free_train, free_test, test_ratio=0.1)

    # Downsample helper
    def downsample(arr, factor=0.5, seed=42):
        """Randomly keep a fraction of the samples."""
        if len(arr) == 0 or factor >= 1.0:
            return arr
        rng = np.random.default_rng(seed)
        n_keep = int(len(arr) * factor)
        idx = rng.choice(len(arr), n_keep, replace=False)
        return arr[idx]

    # Build sets by category
    def build_category_sets(train_list, test_list, col, down_free=0.5):
        train = concat_from_list(train_list, col) if train_list else np.array([])
        test = concat_from_list(test_list, col) if test_list else np.array([])

        # If this is the free category, downsample train
        if train_list is free_train and down_free < 1.0:
            train = downsample(train, factor=down_free)

        return train, test

    def build_sets(col, down_free=0.5):
        free_tr, free_te = build_category_sets(free_train, free_test_cut, col, down_free)
        palp_tr, palp_te = build_category_sets(palp_train, palp_test, col)
        tens_tr, tens_te = build_category_sets(tens_train, tens_test, col)

        train = np.concatenate([free_tr, palp_tr, tens_tr])
        test = np.concatenate([free_te, palp_te, tens_te])

        return train, test, {
            "free": (len(free_tr), len(free_te)),
            "palp": (len(palp_tr), len(palp_te)),
            "tens": (len(tens_tr), len(tens_te)),
        }

    # Stack arrays + track category counts
    X_train, X_test, y_train, y_test = [], [], [], []
    category_counts = {"free": [0, 0], "palp": [0, 0], "tens": [0, 0]}

    for col in input_cols:
        tr, te, counts = build_sets(col, down_free=0.5)
        X_train.append(tr)
        X_test.append(te)
        if col == input_cols[0]:
            for k in category_counts:
                category_counts[k][0] = counts[k][0]
                category_counts[k][1] = counts[k][1]

    for col in target_cols:
        tr, te, _ = build_sets(col)
        y_train.append(tr)
        y_test.append(te)

    X_train = np.stack(X_train, axis=1)
    X_test = np.stack(X_test, axis=1)
    y_train = np.stack(y_train, axis=1)
    y_test = np.stack(y_test, axis=1)

    # Train/val split
    def train_val_split(X, y, val_split=0.22):
        split_idx = int(len(X) * (1 - val_split))
        return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]

    X_train, y_train, X_val, y_val = train_val_split(X_train, y_train, val_split=0.11)

    # Print splits
    n_train, n_val, n_test = len(X_train), len(X_val), len(X_test)
    total = n_train + n_val + n_test

    def fmt_split(name, n, free, tens, palp):
        minutes = n * 0.01 * (1 / 60)
        return (
            f"{name}: {n} samples ({minutes:.1f} min) "
            f"{free/n*100:.1f}% free, {tens/n*100:.1f}% tens, {palp/n*100:.1f}% palp"
        )

    # Extract category counts
    free_train_count, free_test_count = category_counts["free"]
    tens_train_count, tens_test_count = category_counts["tens"]
    palp_train_count, palp_test_count = category_counts["palp"]

    # Split validation proportionally from training
    def split_val(train_count):
        return int(train_count * (n_val / (n_train + n_val))) if (n_train + n_val) > 0 else 0

    free_val = split_val(free_train_count)
    tens_val = split_val(tens_train_count)
    palp_val = split_val(palp_train_count)

    # Adjusted train after taking validation
    free_train_count -= free_val
    tens_train_count -= tens_val
    palp_train_count -= palp_val

    print(f"\nOverall split:")
    print(f"  Train: {n_train/total*100:.1f}%")
    print(f"  Val:   {n_val/total*100:.1f}%")
    print(f"  Test:  {n_test/total*100:.1f}%")

    print(fmt_split("Train", n_train, free_train_count, tens_train_count, palp_train_count))
    print(fmt_split("Val", n_val, free_val, tens_val, palp_val))
    print(fmt_split("Test", n_test, free_test_count, tens_test_count, palp_test_count))

    # Create datasets
    train_dataset = WindowDataset(X_train, y_train, lookback=100, stride=5)
    f_mean, f_std = train_dataset.force_mean, train_dataset.force_std

    val_dataset = WindowDataset(
        X_val, y_val, lookback=100, stride=1, f_mean=f_mean, f_std=f_std
    )
    test_dataset = WindowDataset(
        X_test, y_test, lookback=100, stride=1, f_mean=f_mean, f_std=f_std
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, f_mean, f_std
