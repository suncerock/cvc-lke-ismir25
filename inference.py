import os
import yaml
from string import Template

import numpy as np
import pandas as pd
import torch

from data.datamodule import LocalKeyDataModule
from data.utils import INDEX_TO_KEY
from models.model import LKEModel


def inference(datamodule: LocalKeyDataModule, model: LKEModel, output_dir: str, label_fps: float, device: str = "cpu"):

    predictions_dir = os.path.join(output_dir, "predictions")
    annotations_dir = os.path.join(output_dir, "annotations")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    datamodule.setup()

    for dataloader_idx, dataloader in enumerate(datamodule.test_dataloader()):
        for batch_idx, batch in enumerate(dataloader):
            with torch.no_grad():
                x, y = batch["x"].to(device), batch["y"].to(device)
                track_name = batch["track_name"][0]

                feature, y_pred = model.predict_step(x)

                y_pred = y_pred.squeeze(0)
                feature = feature.squeeze(0)
                y = y.squeeze(0)

                y_pred = torch.softmax(y_pred, dim=0)

                y_pred = y_pred.detach().cpu().numpy().T
                y = y.detach().cpu().numpy()

                np.save(os.path.join(predictions_dir, f"{track_name}.npy"), y_pred)
                np.save(os.path.join(annotations_dir, f"{track_name}.npy"), y)

                count = np.count_nonzero(y != -1)
                correct = np.count_nonzero(y_pred.argmax(axis=-1)[y != -1] == y[y!= -1])
                accuracy = correct / count * 100
                print(f"{track_name}: {accuracy:.2f}%")

def inference_method(config_path: str, checkpoint_dir: str, output_dir: str, dataset_dir: str, repo_dir: str, device: str = "cpu"):
    with open(config_path) as f:
        template = Template(f.read())
        config = template.substitute(DATASET_DIR=dataset_dir, REPO_DIR=repo_dir)
        config = yaml.safe_load(config)
    label_fps = config["data"]["label_fps"]

    config["data"]["num_workers"] = 0  # We do inference track by track
    datamodule = LocalKeyDataModule(**config["data"])
    
    model = LKEModel(**config["model"])
    model.to(device)
    model.eval()    

    for runs in os.listdir(checkpoint_dir):
        run_dir = os.path.join(checkpoint_dir, runs)
        for checkpoint in os.listdir(run_dir):
            if not checkpoint.endswith(".ckpt"):
                continue
            checkpoint_path = os.path.join(run_dir, checkpoint)
            model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True)["state_dict"])
            
            inference(
                datamodule=datamodule,
                model=model,
                output_dir=os.path.join(output_dir, runs, checkpoint.replace(".ckpt", "")),
                label_fps=label_fps,
                device=device
            )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, default="../datasets")
    parser.add_argument("--repo_dir", type=str, default="./")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    inference_method(
        config_path=args.config_path,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        dataset_dir=args.dataset_dir,
        repo_dir=args.repo_dir,
        device=args.device
   )
