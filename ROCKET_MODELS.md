# ROCKET Family Regression Models for Shaft Force Sensing

## Overview

This document explains the ROCKET family regression models that have been integrated into the shaft force sensing project as an alternative to LTC (Liquid Time-Constant) models.

## What are ROCKET Models?

**ROCKET** stands for **Random Convolutional Kernel Transform**. These are efficient time series regression methods from the [aeon-toolkit](https://www.aeon-toolkit.org/) that use random convolutional kernels to extract features from sequential data.

### Key Characteristics:

- ✅ **Fast Training**: No iterative optimization required; direct feature transformation
- ✅ **Competitive Accuracy**: Often matches or exceeds deep learning approaches
- ✅ **Scikit-learn Compatible**: Works with standard ML pipelines
- ✅ **Minimal Hyperparameter Tuning**: Fewer parameters to optimize
- ✅ **Parallelizable**: Built-in support for multi-core computation
- ✅ **Memory Efficient**: Lower memory footprint than neural networks

## Available ROCKET Models

### 1. RocketRegressor
**Base ROCKET implementation with single-scale kernels**

```python
from shaft_force_sensing.models import LitRocket

model = LitRocket(
    d_input=20,              # Number of input features
    d_output=3,              # 3D force output (fx, fy, fz)
    num_kernels=10000        # Number of random convolutional kernels
)
```

**Characteristics:**
- Standard ROCKET with fixed kernel scales
- Good baseline performance
- Typical kernel count: 10,000-20,000
- Training time: Fast (seconds to minutes on modern hardware)

### 2. MultiRocketRegressor  
**Multi-scale ROCKET with kernels at different temporal scales**

```python
from shaft_force_sensing.models import LitMultiRocket

model = LitMultiRocket(
    d_input=20,
    d_output=3,
    num_kernels=5000         # Often fewer kernels needed due to multi-scale
)
```

**Characteristics:**
- Uses multiple scales of convolutional kernels
- Better captures temporal patterns at different resolutions
- **Recommended for shaft force sensing** - captures both fine and coarse force variations
- Typical kernel count: 5,000-10,000
- Training time: Moderate (slightly slower than ROCKET)

### 3. HydraRegressor
**Ensemble approach combining ROCKET with multiple Ridge regressors**

```python
from shaft_force_sensing.models import LitHydra

model = LitHydra(
    d_input=20,
    d_output=3,
    num_kernels=2000         # Fewer kernels in ensemble form
)
```

**Characteristics:**
- Combines ROCKET features with Ridge regression ensemble
- Typically highest accuracy
- Slower training and prediction than base ROCKET
- Typical kernel count: 2,000-5,000
- Training time: Moderate to slow (but still much faster than neural networks)

## PyTorch Lightning Integration

All ROCKET models have been wrapped in PyTorch Lightning classes for integration with the existing training pipeline.

### Training ROCKET Models

```python
from torch.utils.data import DataLoader
from shaft_force_sensing.models import LitMultiRocket

# Initialize model
model = LitMultiRocket(
    d_input=20,
    d_output=3,
    num_kernels=5000
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256)

# Train model (fits directly to data without iteration)
model.train_rocket(train_loader, val_loader)

# Make predictions
import torch
with torch.no_grad():
    for batch in val_loader:
        x, y, mask = batch
        predictions = model(x)
```

## Comparison with LTC Models

| Aspect | ROCKET | LTC |
|--------|--------|-----|
| Training Approach | Direct feature transform | Recurrent neural network |
| Training Time | Fast (seconds) | Slow (minutes) |
| Dependencies | scikit-learn, aeon | PyTorch, NCP library |
| Hyperparameters | ~3-4 | ~10+ |
| Memory Usage | Low | Medium-High |
| Interpretability | Moderate | Low |
| Scalability | Excellent | Good |
| Multi-scale Temporal Analysis | MultiRocket | Limited |

## Why Switch from LTC to ROCKET?

1. **Speed**: ROCKET models train orders of magnitude faster
2. **Efficiency**: Lower memory footprint, simpler dependencies
3. **Reliability**: No convergence issues, deterministic training
4. **Flexibility**: Easy to ensemble and compare multiple variants
5. **Effectiveness**: Competitive or superior accuracy for time series regression
6. **Robustness**: Less sensitive to hyperparameter choices

## Installation

Update your environment with the required dependencies:

```bash
pip install -r requirements.txt
```

Or install aeon-toolkit directly:

```bash
pip install aeon-toolkit>=0.11.0
```

## Usage Example - Full Training Pipeline

```python
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch

from shaft_force_sensing import ForceSensingDataset
from shaft_force_sensing.models import LitRocket, LitMultiRocket, LitHydra

# 1. Load data
dataset = ForceSensingDataset(data_dir, input_cols, target_cols)
train_size = int(0.8 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256)

# 2. Compare different ROCKET variants
models = {
    'ROCKET': LitRocket(d_input=20, d_output=3, num_kernels=10000),
    'MultiROCKET': LitMultiRocket(d_input=20, d_output=3, num_kernels=5000),
    'Hydra': LitHydra(d_input=20, d_output=3, num_kernels=2000),
}

# 3. Train all models
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.train_rocket(train_loader, val_loader)
    
    # Evaluate
    with torch.no_grad():
        all_preds, all_gts = [], []
        for batch in val_loader:
            x, y, _ = batch
            pred = model(x)
            all_preds.append(pred.cpu().numpy())
            all_gts.append(y.cpu().numpy())
    
    y_pred = np.vstack(all_preds).squeeze()
    y_gt = np.vstack(all_gts).squeeze()
    
    mse = mean_squared_error(y_gt, y_pred)
    r2 = r2_score(y_gt, y_pred)
    
    print(f"  MSE: {mse:.6f}, R²: {r2:.4f}")
```

## Reference Training Notebook

See [train_rocket.ipynb](../notebooks/train_rocket.ipynb) for a complete example with:
- Data loading and preparation
- Training multiple ROCKET variants
- Evaluation and comparison
- Visualization of results

## Performance Recommendations

For your shaft force sensing application:

1. **Start with**: **MultiRocketRegressor** with 5,000 kernels
   - Good accuracy for capturing force dynamics
   - Reasonable training time
   - Balances performance and efficiency

2. **For maximum accuracy**: **HydraRegressor** with 2,000-3,000 kernels
   - Best predictive performance
   - Still trains in seconds
   - Ensemble approach is robust

3. **For speed**: **RocketRegressor** with 10,000+ kernels
   - Fastest training
   - Competitive accuracy with simple architecture
   - Good baseline for quick iterations

## Hyperparameter Tuning

For ROCKET models, there are few hyperparameters to tune:

### num_kernels
- Controls model capacity (more kernels = potentially better accuracy, slower training)
- Recommended ranges:
  - ROCKET: 5,000-20,000
  - MultiROCKET: 3,000-10,000
  - Hydra: 1,000-5,000

### Other parameters
- `normalise`: Whether to normalize inputs (default: True, recommended)
- `random_state`: For reproducibility (set to any integer)
- `n_jobs`: Number of parallel jobs (default: -1 for all available cores)

## File Structure

```
shaft_force_sensing/
├── models/
│   ├── __init__.py                 # Exports all models
│   ├── rocket_models.py            # Base ROCKET implementations
│   ├── rocket_lightning.py         # PyTorch Lightning wrappers
│   ├── lightning.py                # (existing) Lightning base classes
│   ├── ltc.py                      # (existing) LTC models
│   └── transformer.py              # (existing) Transformer models
└── notebooks/
    ├── train_rocket.ipynb          # ROCKET training examples
    ├── train_ltc.ipynb             # (existing) LTC training
    └── train_transformer.ipynb     # (existing) Transformer training
```

## References

- **aeon-toolkit**: https://www.aeon-toolkit.org/
- **ROCKET Paper**: [ROCKET: Exceptionally fast and accurate time series classification](https://arxiv.org/abs/1910.13558)
- **MultiROCKET**: [MultiROCKET: Explainable fast time series classification](https://arxiv.org/abs/2102.00981)
- **Hydra**: [Hydra: An Ensemble of Diverse Explainable Time Series Classifiers](https://arxiv.org/abs/2208.10218)

## Troubleshooting

### Installation issues with aeon-toolkit

If you encounter build errors:

```bash
# Try installing with pre-built wheels
pip install --upgrade aeon-toolkit

# Or specify a compatible version
pip install aeon-toolkit==0.11.0
```

### Memory issues with large datasets

If running out of memory:
- Reduce `num_kernels`
- Reduce `batch_size` in DataLoader
- Use fewer cores with `n_jobs=2` or `n_jobs=4`

### Slow training despite "fast ROCKET"

- Ensure data preprocessing is cached (not recomputed each epoch)
- Check that `n_jobs` is set to use multiple cores
- Verify you're using the correct input shape: (batch, seq_len, features)

## Questions or Issues?

For issues specific to:
- **aeon-toolkit**: https://github.com/aeon-toolkit/aeon
- **ROCKET implementations**: See docstrings in `rocket_models.py`
- **Integration**: Check `rocket_lightning.py` for PyTorch Lightning wrapping
