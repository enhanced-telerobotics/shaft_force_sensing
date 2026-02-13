"""Transformer model architecture for force prediction."""

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model.
    
    Parameters
    ----------
    d_model : int
        Dimension of the model
    max_len : int, optional
        Maximum sequence length (default: 500)
    """

    def __init__(self, d_model, max_len=500):
        """Initialize positional encoding."""
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Apply positional encoding to input.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch, seq_len, d_model]
        
        Returns
        -------
        torch.Tensor
            Tensor with positional encoding added
        """
        x = x + self.pe[:, : x.size(1)]
        return x


class TransformerModel(nn.Module):
    """Transformer-based force prediction model.
    
    Parameters
    ----------
    d_input : int
        Number of input features
    d_output : int, optional
        Output dimension for force (default: 3)
    d_model : int, optional
        Dimension of the transformer model (default: 64)
    nhead : int, optional
        Number of attention heads (default: 8)
    num_layers : int, optional
        Number of transformer encoder layers (default: 3)
    """

    def __init__(
        self,
        d_input,
        d_output=3,
        d_model=64,
        nhead=8,
        num_layers=3,
    ):
        """Initialize the transformer model."""
        super().__init__()

        self.input_proj = nn.Linear(d_input, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Shared intermediate representation
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_output)
        )

    def forward(self, x, mask):
        """Forward pass of the model.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch, seq_len, input_size]
        mask : torch.Tensor
            Mask tensor of shape [batch, seq_len] (True for padding, False for valid data)
        
        Returns
        -------
        torch.Tensor
            Force predictions of shape [batch, output_size]
        """
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        x = self.head(x[:, -1, :])

        return x
