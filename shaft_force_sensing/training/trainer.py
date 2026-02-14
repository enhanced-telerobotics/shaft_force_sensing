import os
from pathlib import Path
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from shaft_force_sensing.models import (
    LitSequenceModel,
    LitTransformer,
    LitLTC,
)
from shaft_force_sensing.training.utils import (
    args_parser,
    prepare_datasets,
    get_input_cols_for_config
)

# Global column definitions (full set)
i_cols = [
    'jaw_position', 'wrist_pitch_position', 'wrist_yaw_position', 'roll_position',
    'wrist_pitch_velocity', 'wrist_yaw_velocity', 'jaw_velocity', 'roll_velocity',
    'wrist_pitch_effort', 'wrist_yaw_effort', 'roll_effort',
    'jaw_effort', 'insertion_effort', 'yaw_effort', 'pitch_effort',
    'tx', 'ty', 'tz', 'fx', 'fy', 'fz'
]
t_cols = ['ati_fx', 'ati_fy', 'ati_fz']


def train_model(
    model: LitSequenceModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_epochs: int = 50,
    save_dir: str = "logs"
) -> None:
    # Set up save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Set up logger and callbacks
    logger = TensorBoardLogger(
        save_dir,
        name=model._get_name())
    early_stop_callback = EarlyStopping(
        monitor="val/loss_epoch",
        min_delta=1e-4,
        patience=3,
        verbose=True,
        mode="min"
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        monitor="val/loss_epoch",
        mode="min",
        save_top_k=1,
        verbose=True,
        filename="best-epoch={epoch:02d}-loss={val/loss_epoch:.4f}",
        auto_insert_metric_name=False
    )

    # Initialize PyTorch Lightning trainer
    trainer = Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    # Train the model
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
    ablation_config = args.get("ablation_config", "Full")

    # Set random seed for reproducibility
    seed_everything(seed)

    # Get input columns for the specified ablation configuration
    ablated_i_cols = get_input_cols_for_config(ablation_config)
    print(f"Using ablation configuration: {ablation_config}")
    print(f"Input columns: {len(ablated_i_cols)}")
    if ablation_config != "Full":
        print(f"Removed columns: {set(i_cols) - set(ablated_i_cols)}")

    # Prepare datasets and dataloaders
    train_set, val_set, scaler = prepare_datasets(
        os.getcwd(), ablated_i_cols, t_cols)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # Initialize model based on specified type
    if model_type == "transformer":
        model_cls = LitTransformer
    elif model_type == "ltc":
        model_cls = LitLTC
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model: LitSequenceModel = model_cls(
        d_input=len(ablated_i_cols),
        d_output=len(t_cols),
        d_hidden=args.get("hidden_size", 64),
        data_mean=scaler.mean_,
        data_std=scaler.scale_,
        **args
    )

    # Train the model
    train_model(model, train_loader, val_loader, max_epochs, save_dir)
