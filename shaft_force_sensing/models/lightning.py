"""PyTorch Lightning module for transformer model training."""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter

from .transformer import TransformerModel
from .ltc import LTCModel
from .lstm import LSTMModel


class LitSequenceModel(pl.LightningModule):
    def __init__(
        self,
        d_input,
        d_output=3,
        d_hidden=64,
        lr=3e-4,
        weight_decay=1e-4,
        lr_scheduler_patience=3,
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
            d_hidden (int, optional): Dimension of the model/hidden layers. Defaults to 64.
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
        self.d_hidden = d_hidden

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
            d_model=self.d_hidden,
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
            d_hidden=self.d_hidden,
        )

        self._hidden_state = None

    def forward(self, x, *args, **kwargs):
        out, hidden = self.model(x, self._hidden_state)
        # TODO: handle hidden state for LTC
        return out


class LitLSTM(LitSequenceModel):
    def __init__(
        self,
        num_layers=1,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.models = nn.ModuleList([
            LSTMModel(
                d_input=self.d_input,
                d_hidden=self.d_hidden,
                num_layers=num_layers,
                dropout=dropout,
            )
            for _ in range(6)
        ])

        self.register_buffer("tao_max", torch.tensor(
            [10.0000, 10.0000, 10.0000, 0.2000, 0.2000, 0.2000]))
        
        self.reset_hidden()

    def setup(self, stage):
        if stage in ("test", "predict"):
            self.reset_hidden()

    def reset_hidden(self):
        self.hiddens = [None] * 6

    def forward(self, pos, vec, lengths):
        x = torch.cat([pos, vec], dim=-1)

        out_list = []
        hidden_list = []
        for idx, model in enumerate(self.models):
            out_i, hidden_i = model(x, lengths, self.hiddens[idx])
            out_list.append(out_i)
            hidden_list.append(hidden_i)

        out = torch.cat(out_list, dim=-1)
        out *= self.tao_max
        return out, hidden_list

    def training_step(self, batch, batch_idx):
        """Training step."""
        pos, vec, tau, _, _, lens = batch
        pred_tau, _ = self(pos, vec, lens)

        loss = self.loss_fn(pred_tau, tau)

        self.log("train/loss", loss, prog_bar=True,
                 logger=True, on_epoch=True, on_step=False)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        pos, vec, tau, force, jaco, lens = batch
        pred_tau, _ = self(pos, vec, lens)

        pred_force = self.calc_force(jaco, tau, pred_tau)

        # Use for free space only
        # Since predicted force in the tool frame
        loss = self.loss_fn(pred_force, force)

        self.log("val/loss", loss, prog_bar=True,
                 logger=True, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        """Test step."""
        pos, vec, tau, force, jaco, lens = batch
        pred_tau, self.hiddens = self(pos, vec, lens)
        
        pred_force = self.calc_force(jaco, tau, pred_tau)

        assert pred_force.size(
            0) == 1, "Test batch size > 1 not supported for force logging."

        logger: SummaryWriter = self.logger.experiment

        # TODOs:
        # Convert to hex10 sensor frame in post-processing
        # Also handle reaction force sign convention in post-processing
        pred = pred_force.squeeze().detach().cpu().tolist()
        gt = force.squeeze().detach().cpu().tolist()

        logger.add_scalars(
            "test/force_0",
            {"pred": pred[0], "gt": gt[0]},
            global_step=batch_idx,
        )
        logger.add_scalars(
            "test/force_1",
            {"pred": pred[1], "gt": gt[1]},
            global_step=batch_idx,
        )
        logger.add_scalars(
            "test/force_2",
            {"pred": pred[2], "gt": gt[2]},
            global_step=batch_idx,
        )

    def calc_force(self, jacobian, tau, pred_tau):
        # F = J^-T * (tau - tau_pred)
        J_inv_T = torch.linalg.inv(jacobian).transpose(-1, -2)
        delta_tau = tau - pred_tau
        wrench = torch.einsum("...ij,...j->...i", J_inv_T, delta_tau)
        return wrench[:, :3]
