"""Model training functions."""

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer

from ..models import LitTransformer


def train_model(train_loader, val_loader, f_mean, f_std, input_cols, max_epochs=30):
    """Train a transformer model for force prediction.
    
    Parameters
    ----------
    train_loader : torch.utils.data.DataLoader
        Training data loader
    val_loader : torch.utils.data.DataLoader
        Validation data loader
    f_mean : np.ndarray
        Mean of force targets for normalization
    f_std : np.ndarray
        Standard deviation of force targets for normalization
    input_cols : list
        List of input columns (used to determine input_size)
    max_epochs : int, optional
        Maximum number of epochs (default: 30)
    
    Returns
    -------
    LitTransformer
        Trained model
    """
    model = LitTransformer(
        input_size=len(input_cols),
        d_model=128,
        nhead=8,
        num_layers=3,
        force_output_size=3,
        lr=3e-4,
        data_mean=f_mean,
        data_std=f_std,
    )

    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=3, verbose=True, mode="min"
    )

    # Checkpoint to save best model
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True,
        filename="best_model",
    )

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        logger=False,
        callbacks=[early_stop_callback, checkpoint_callback],
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader, val_loader)

    # Load the best model after training
    best_model_path = checkpoint_callback.best_model_path
    model = LitTransformer.load_from_checkpoint(best_model_path)

    return model
