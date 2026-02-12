"""Shaft Force Sensing - Transformer-based force prediction."""

__version__ = "0.1.0"
__author__ = "Erie Lab"

from .data import ForceSensingDataset
from .models import TransformerModel, PositionalEncoding, LitTransformer
from .training import train_model

__all__ = [
    "ForceSensingDataset",
    "TransformerModel",
    "PositionalEncoding",
    "LitTransformer",
]
