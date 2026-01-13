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
    input_size : int
        Number of input features
    d_model : int
        Dimension of the transformer model
    nhead : int
        Number of attention heads
    num_layers : int
        Number of transformer encoder layers
    force_output_size : int, optional
        Output dimension for force (default: 3)
    fc_hidden_size : int, optional
        Hidden size for fully connected layers (default: 64)
    data_mean : np.ndarray, optional
        Mean for data normalization
    data_std : np.ndarray, optional
        Standard deviation for data normalization
    """

    def __init__(
        self,
        input_size,
        d_model,
        nhead,
        num_layers,
        force_output_size=3,
        fc_hidden_size=64,
        data_mean=None,
        data_std=None,
    ):
        """Initialize the transformer model."""
        super(TransformerModel, self).__init__()

        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.0,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Shared intermediate representation
        self.shared_fc = nn.Sequential(
            nn.Linear(d_model, fc_hidden_size),
            nn.ReLU(),
        )

        # Force regression head
        self.force_head = nn.Linear(fc_hidden_size, force_output_size)

        # Register dataset distribution as buffer
        if data_mean is not None:
            self.register_buffer("data_mean", torch.from_numpy(data_mean))

        if data_std is not None:
            self.register_buffer("data_std", torch.from_numpy(data_std))

    def forward(self, x):
        """Forward pass of the model.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch, seq_len, input_size]
        
        Returns
        -------
        torch.Tensor
            Force predictions of shape [batch, 3]
        """
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        last_step = x[:, -1, :]  # [batch, d_model]

        shared = self.shared_fc(last_step)
        force_out = self.force_head(shared)

        return force_out
