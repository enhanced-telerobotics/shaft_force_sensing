"""Shaft Force Sensing - Transformer-based force prediction."""

__version__ = "0.1.0"
__author__ = "Erie Lab"

from .data import WindowDataset, assemble_loaders
from .models import TransformerModel, PositionalEncoding, LitTransformer
from .training import train_model
from .evaluation import test_model, evaluate_results, save_force_results
from .visualization import (
    plot_r2_scatter,
    plot_force_comparison_3x3,
    plot_force_comparison_stacked,
    plot_time_trace_per_axis,
)

__all__ = [
    "WindowDataset",
    "assemble_loaders",
    "TransformerModel",
    "PositionalEncoding",
    "LitTransformer",
    "train_model",
    "test_model",
    "evaluate_results",
    "save_force_results",
    "plot_r2_scatter",
    "plot_force_comparison_3x3",
    "plot_force_comparison_stacked",
    "plot_time_trace_per_axis",
]
