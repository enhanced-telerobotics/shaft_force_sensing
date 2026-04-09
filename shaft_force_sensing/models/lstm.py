"""
LSTM model architecture for force prediction.
Adopted from https://github.com/vu-maple-lab/dvrk_force_estimation/blob/2f378ece9bea4d5805b205c1d78760a7820d07fd/indirect_method/network.py#L39
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMModel(nn.Module):
	"""LSTM-based force prediction model.

	Parameters
	----------
	d_input : int
		Number of input features
	d_hidden : int, optional
		Hidden dimension of the LSTM (default: 64)
	num_layers : int, optional
		Number of stacked LSTM layers (default: 2)
	dropout : float, optional
		Dropout between LSTM layers (default: 0.1)
	"""

	def __init__(
		self,
		d_input:int,
		d_hidden:int=64,
		num_layers:int=1,
		dropout:float=0.1,
	) -> None:
		"""Initialize the LSTM model.
		"""
		super().__init__()
		self.num_layers = num_layers
		self.d_hidden = d_hidden

		self.lstm = nn.LSTM(
			input_size=d_input,
			hidden_size=d_hidden,
			num_layers=num_layers,
			dropout=dropout if num_layers > 1 else 0.0,
			batch_first=True,
		)

		self.hn = None

		self.head = nn.Sequential(
			nn.Linear(d_hidden, d_hidden//2),
			nn.ReLU(),
			nn.Linear(d_hidden//2, 1),
			nn.Tanh(),
		)

	def forward(
			self, 
			x: torch.Tensor, 
			lengths: torch.Tensor,
			hx: tuple = None) -> torch.Tensor:
		"""Forward pass of the model.

		Parameters
		----------
		x : torch.Tensor
			Input tensor of shape [batch, seq_len, input_size]
		lengths : torch.Tensor
			Valid sequence lengths for each batch sample.
			If provided, padded tokens are skipped via pack/unpack.
		hx : tuple, optional
			Initial hidden and cell states for the LSTM, each of shape [num_layers, batch, hidden_size].
			If None, initialized to zeros. (default: None)

		Returns
		-------
		torch.Tensor
			Force predictions of shape [batch, output_size]
		tuple
			Final hidden and cell states of the LSTM, each of shape [num_layers, batch, hidden_size]
		"""
		# lengths = lengths.to(dtype=torch.long)
		# lengths = lengths.clamp(min=1, max=x.size(1))

		# packed = pack_padded_sequence(
		# 	x,
		# 	lengths=lengths.detach().cpu(),
		# 	batch_first=True,
		# 	enforce_sorted=False,
		# )
		# packed_out, hn = self.lstm(packed, hx)
		# unpacked, _ = pad_packed_sequence(
		# 	packed_out,
		# 	batch_first=True,
		# 	total_length=x.size(1),
		# )

		# batch_idx = torch.arange(unpacked.size(0), device=unpacked.device)
		# last_valid = lengths.to(unpacked.device) - 1
		# x = unpacked[batch_idx, last_valid, :]
		x, hn = self.lstm(x, hx)
		x = x[:, -1, :]
		x = self.head(x)

		return x, hn
