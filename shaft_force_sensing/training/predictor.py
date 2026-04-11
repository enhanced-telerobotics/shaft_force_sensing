from pathlib import Path

from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything, Trainer
from lightning.pytorch.loggers import TensorBoardLogger

import shaft_force_sensing.models
from shaft_force_sensing.models import LitSequenceModel
from shaft_force_sensing.training.utils import (
    args_parser,
    prepare_test_dataset,
    load_model,
)


def predict_model(model: LitSequenceModel, save_dir: Path, data_root: Path) -> None:
    finetune = model.hparams.get("finetune", False)
    test_sets, batch_size = prepare_test_dataset(
        data_root,
        model._get_name(),
        finetune=finetune,
        ablations=model.hparams.get("ablations", None),
        model_idx=model.hparams.get("model_idx", 0),
        sequence_length=model.hparams.get("sequence_length", 100),
    )
    test_loaders = {
        group: DataLoader(dset, batch_size=batch_size, shuffle=False)
        for group, dset in test_sets.items()
    }

    for group, loader in test_loaders.items():
        logger = TensorBoardLogger(
            save_dir,
            name="test" if not finetune else "teleop_test",
            version=group,
        )
        Trainer(logger=logger).test(model=model, dataloaders=loader)


if __name__ == "__main__":
    args = args_parser()
    print(args)
    seed_everything(args["seed"])

    assert args["save_dir"] is not None, "--save_dir is required for prediction."
    save_dir = Path(args["save_dir"])
    assert save_dir.is_dir(), f"Save directory {save_dir} does not exist."

    model = load_model(save_dir, **args)
    predict_model(model=model, save_dir=save_dir,
                  data_root=Path().cwd() / "data")
