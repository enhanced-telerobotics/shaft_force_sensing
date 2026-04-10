from pathlib import Path
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor
)
from lightning.pytorch.loggers import TensorBoardLogger

import shaft_force_sensing.models
from shaft_force_sensing.models import (
    LitSequenceModel,
    LitTransformer,
    LitLTC,
    LitLSTM,
)
from shaft_force_sensing.training.utils import (
    args_parser,
    prepare_datasets,
)


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
        monitor="val/loss",
        min_delta=1e-4,
        patience=10,
        verbose=True,
        mode="min"
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        verbose=True,
        filename="best-epoch={epoch:02d}-loss={val/loss:.4f}",
        auto_insert_metric_name=False
    )
    lr_callback = LearningRateMonitor(
        logging_interval="epoch"
    )

    # Initialize PyTorch Lightning trainer
    trainer = Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[
            early_stop_callback,
            checkpoint_callback,
            lr_callback
        ],
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
    finetune = args.get("finetune", False)
    save_dir = args["save_dir"]
    model_dir = args["model_dir"]

    # Set random seed for reproducibility
    seed_everything(seed)

    # Prepare datasets and dataloaders based on model type
    train_set, val_set, scaler = prepare_datasets(
        Path().cwd() / "data",
        model_type,
        finetune=finetune,
        ablations=args.get("ablations", None),
        model_idx=args.get("model_idx", 0),
        stride=args.get("stride", 5),
        sequence_length=args.get("sequence_length", 100),
    )
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    if not finetune:
        # Initialize model based on specified type
        if model_type == "transformer" or model_type == "ltc":
            model = LitTransformer(
                d_input=train_set[0][0].shape[1],
                d_output=train_set[0][1].shape[0],
                d_hidden=args.get("hidden_size", 64),
                data_mean=scaler.mean_.tolist(),
                data_std=scaler.scale_.tolist(),
                **args
            )
        elif model_type == "lstm":
            model = LitLSTM(
                d_input=train_set[0][0].shape[1] * 2,
                d_hidden=args.get("hidden_size", 128),
                **args
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    else:
        model_dir = Path(model_dir)
        assert model_dir.is_dir(), f"Model directory {model_dir} does not exist."

        # Load model from checkpoint for fine-tuning
        for p in model_dir.iterdir():
            if p.name in shaft_force_sensing.models.__all__:
                model_cls = p.name
                break
        assert model_cls is not None, "Model name not found in checkpoint directory."

        model: LitSequenceModel = eval(model_cls).load_from_checkpoint(
            sorted(model_dir.glob("best*.ckpt"))[-1],
            **args
        )


    # Train the model
    train_model(model, train_loader, val_loader, max_epochs, save_dir)
