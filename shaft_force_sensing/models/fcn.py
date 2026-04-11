"""Fully connected model architecture for force prediction."""

import torch
import torch.nn as nn


class FCNModel(nn.Module):
	"""Feed-forward force prediction model.

	The model always uses the last timestep when a sequence tensor is passed in,
	which keeps it compatible with the existing sequence-based datasets without
	changing their implementation.
	"""

	def __init__(
		self,
		d_input,
		d_output=3,
		d_hidden=64,
		num_layers=3,
		dropout=0.1,
	):
		super().__init__()

		if num_layers < 1:
			raise ValueError("num_layers must be at least 1")

		layers = []
		in_features = d_input
		for _ in range(num_layers - 1):
			layers.extend(
				[
					nn.Linear(in_features, d_hidden),
					nn.ReLU(),
					nn.Dropout(dropout),
				]
			)
			in_features = d_hidden

		self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
		self.head = nn.Linear(in_features, d_output)

	def forward(self, x, mask=None):
		"""Forward pass.

		Parameters
		----------
		x : torch.Tensor
			Input tensor of shape [batch, seq_len, input_size] or [batch, input_size].
		mask : torch.Tensor, optional
			Unused, kept for interface compatibility with sequence models.
		"""
		if x.dim() == 3:
			x = x[:, -1, :]
		elif x.dim() != 2:
			raise ValueError(
				f"Expected a 2D or 3D input tensor, got shape {tuple(x.shape)}"
			)

		x = self.backbone(x)
		return self.head(x)
