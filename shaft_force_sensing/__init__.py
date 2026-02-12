"""Shaft Force Sensing - Transformer-based force prediction."""

__version__ = "0.1.0"
__author__ = "Erie Lab"

from .data import ForceSensingDataset
from .models import TransformerModel, PositionalEncoding, LitTransformer
from .training import train_model
from .evaluation import test_model, evaluate_results, save_force_results

__all__ = [
    "ForceSensingDataset",
    "TransformerModel",
    "PositionalEncoding",
    "LitTransformer",
    "train_model",
    "test_model",
    "evaluate_results",
]
