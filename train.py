import argparse
import yaml
from string import Template

import numpy as np
import wandb
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from data.datamodule import LocalKeyDataModule
from models.model import LKEModel

def train(config_path: str, dataset_dir: str, repo_dir: str, run_id: str):
    with open(config_path) as f:
        template = Template(f.read())
        config = template.substitute(DATASET_DIR=dataset_dir, REPO_DIR=repo_dir)
        config = yaml.safe_load(config)

    datamodule = LocalKeyDataModule(**config["data"])
    datamodule.setup()
    log_path = config["trainer"]["args"].pop("log_path")

    model = LKEModel(**config["model"])
    print(sum([np.prod(param.shape) for param in model.parameters()]))

    bestl_checkpoint = ModelCheckpoint(
        dirpath=log_path.format(run_id=run_id), **config["trainer"]["best_checkpoint"]
    )

    periodic_checkpoint = ModelCheckpoint(
        dirpath=log_path.format(run_id=run_id), **config["trainer"]["periodic_checkpoint"]
    )

    trainer = Trainer(
        **config["trainer"]["args"],
        # logger=WandbLogger(**config["trainer"]["logger"]),
        logger=False,
        callbacks=[bestl_checkpoint, periodic_checkpoint]
    )

    trainer.fit(model, train_dataloaders=datamodule.train_dataloader(), val_dataloaders=datamodule.val_dataloader())
    trainer.test(dataloaders=datamodule.test_dataloader())

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, default="../datasets")
    parser.add_argument("--repo_dir", type=str, default="./")
    parser.add_argument("--run_id", type=str, default="1")

    args = parser.parse_args()
    train(
        config_path=args.config_path,
        dataset_dir=args.dataset_dir,
        repo_dir=args.repo_dir,
        run_id=args.run_id
    )
