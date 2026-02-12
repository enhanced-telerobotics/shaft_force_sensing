"""PyTorch Lightning module for transformer model training."""

import torch
import torch.nn as nn
import pytorch_lightning as pl

from .transformer import TransformerModel


class LitTransformer(pl.LightningModule):
    """PyTorch Lightning module for Transformer-based force prediction.
    
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
    lr : float, optional
        Learning rate (default: 3e-4)
    l1_lambda : float, optional
        L1 regularization coefficient (default: 1e-4)
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
        lr=3e-4,
        l1_lambda=1e-4,
        data_mean=None,
        data_std=None,
    ):
        """Initialize the Lightning module."""
        super().__init__()
        self.save_hyperparameters()

        self.model = TransformerModel(
            input_size,
            d_model,
            nhead,
            num_layers,
            force_output_size,
            data_mean=data_mean,
            data_std=data_std,
        )

        self.loss_fn = nn.MSELoss()
        self.lr = lr
        self.l1_lambda = l1_lambda

    def forward(self, x, mask=None):
        """Forward pass."""
        return self.model(x, mask)

    def training_step(self, batch, batch_idx):
        """Training step."""
        x, gt, mask = batch
        pred = self(x, mask)

        mse_loss = self.loss_fn(pred, gt)

        # L1 regularization
        l1_reg = sum(torch.sum(torch.abs(p)) for p in self.parameters())

        loss = mse_loss + self.l1_lambda * l1_reg

        self.log("train_loss", loss, prog_bar=True,
                 logger=True, on_epoch=True, on_step=True)
        self.log("train_mse_loss", mse_loss)
        self.log("train_l1_reg_loss", self.l1_lambda * l1_reg)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, gt, mask = batch
        pred = self(x, mask)

        loss = self.loss_fn(pred, gt)

        self.log("val_loss", loss, prog_bar=True,
                 logger=True, on_epoch=True, on_step=True)

    def test_step(self, batch, batch_idx):
        """Test step."""
        x, gt, mask = batch
        pred = self(x, mask)

        loss = self.loss_fn(pred, gt)

        self.log("test_loss", loss)

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.lr)
