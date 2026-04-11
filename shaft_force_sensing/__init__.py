"""Shaft Force Sensing - Transformer-based force prediction."""

__version__ = "0.1.0"
__author__ = "Erie Lab"

from .data import SensorDataset, TorqueDataset
from .models import (
    LitTransformer,
    LitFCN,
    LitLTC,
    LitLSTM
)

__all__ = [
    "SensorDataset",
    "TorqueDataset",
    "LitTransformer",
    "LitFCN",
    "LitLTC",
    "LitLSTM",
]
