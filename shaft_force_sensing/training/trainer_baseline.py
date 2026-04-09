from pathlib import Path
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from shaft_force_sensing.models import LitLSTM
from shaft_force_sensing.training.utils import (
    args_parser,
    prepare_baseline_dataset
)


def train_model(
    model: LitLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_epochs: int = 50,
    save_dir: str = "logs",
) -> None:
    """Train the baseline model with Lightning callbacks."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger = TensorBoardLogger(save_dir, name=model._get_name())
    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        min_delta=1e-4,
        patience=10,
        verbose=True,
        mode="min",
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        verbose=True,
        filename="best-epoch={epoch:02d}-loss={val/loss:.4f}",
        auto_insert_metric_name=False,
    )
    lr_callback = LearningRateMonitor(logging_interval="epoch")

    trainer = Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[early_stop_callback, checkpoint_callback, lr_callback],
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    # Parse command-line arguments
    args = args_parser()
    print(args)
    seed = args["seed"]
    batch_size = args["batch_size"]
    max_epochs = args["max_epochs"]
    model_type = args["model_type"]
    save_dir = args["save_dir"]

    # Set random seed for reproducibility
    seed_everything(seed)

    train_set, val_set = prepare_baseline_dataset(
        Path().cwd() / "data" / "Automated",
        stride=args.get("stride", 10),
        sequence_length=args.get("sequence_length", 1000),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
    )

    model = LitLSTM(
        d_input=12,
        d_hidden=args.get("hidden_size", 128)
    )

    train_model(
        model,
        train_loader,
        val_loader,
        max_epochs,
        save_dir,
    )
