"""PyTorch Lightning module for transformer model training."""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter

from .transformer import TransformerModel
from .ltc import LTCModel


class LitSequenceModel(pl.LightningModule):
    def __init__(
        self,
        d_input,
        d_output=3,
        d_model=64,
        lr=3e-4,
        weight_decay=1e-4,
        lr_scheduler_patience=2,
        lr_scheduler_factor=0.1,
        lr_scheduler_min_lr=1e-6,
        data_mean:list=None,
        data_std:list=None,
        **kwargs
    ):
        """
        Initialize the Lightning module for shaft force sensing model.
        Args:
            d_input (int): Dimension of input features.
            d_output (int, optional): Dimension of output features. Defaults to 3.
            d_model (int, optional): Dimension of the model/hidden layers. Defaults to 64.
            lr (float, optional): Learning rate for the optimizer. Defaults to 3e-4.
            weight_decay (float, optional): Weight decay for the optimizer. Defaults to 1e-4.
            data_mean (list, optional): Mean values for dataset normalization. 
                If provided, registered as a buffer. Defaults to None.
            data_std (list, optional): Standard deviation values for dataset normalization. 
                If provided, registered as a buffer. Defaults to None.
        """
        super().__init__()
        self.save_hyperparameters()

        # Initialize model and optimizer parameters
        self.d_input = d_input
        self.d_output = d_output
        self.d_model = d_model

        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_min_lr = lr_scheduler_min_lr

        self.loss_fn = nn.MSELoss()

        # Register dataset distribution as buffer
        if data_mean is not None:
            self.register_buffer("data_mean", torch.tensor(data_mean))

        if data_std is not None:
            self.register_buffer("data_std", torch.tensor(data_std))

    def forward(self, x):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        """Training step."""
        x, gt, mask = batch
        pred = self(x, mask)

        loss = self.loss_fn(pred, gt)

        self.log("train/loss", loss, prog_bar=True,
                 logger=True, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, gt, mask = batch
        pred = self(x, mask)

        loss = self.loss_fn(pred, gt)

        self.log("val/loss", loss, prog_bar=True,
                 logger=True, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        """Test step."""
        x, gt, mask = batch
        pred = self(x, mask)

        logger: SummaryWriter = self.logger.experiment
        batch_size = self.trainer.test_dataloaders.batch_size

        for idx in range(pred.size(0)):
            for dim in range(pred.size(1)):
                logger.add_scalars(
                    f"test/force_{dim}",
                    {"pred": pred[idx, dim], "gt": gt[idx, dim]},
                    global_step=batch_idx * batch_size + idx
                )

    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.lr_scheduler_patience,
            factor=self.lr_scheduler_factor,
            min_lr=self.lr_scheduler_min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


class LitTransformer(LitSequenceModel):
    def __init__(
        self,
        nhead=8,
        num_layers=3,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.model = TransformerModel(
            d_input=self.d_input,
            d_output=self.d_output,
            d_model=self.d_model,
            nhead=nhead,
            num_layers=num_layers,
        )

    def forward(self, x, mask=None):
        return self.model(x, mask)


class LitLTC(LitSequenceModel):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.model = LTCModel(
            d_input=self.d_input,
            d_output=self.d_output,
            d_hidden=self.d_model,
        )

        self._hidden_state = None

    def forward(self, x, *args, **kwargs):
        out, hidden = self.model(x, self._hidden_state)
        # TODO: handle hidden state for LTC
        return out
