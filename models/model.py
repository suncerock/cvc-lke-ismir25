import torch
import torch.nn as nn
import torchmetrics
from lightning import LightningModule

from .features import ALL_FEATURES
from .backbone import ALL_BACKBONES
from .eval import LKEMetric

LKE_NUM_CLASSES = 24


class LKEModel(LightningModule):
    def __init__(
        self,
        feature: dict,
        backbone: dict,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.feature_layer = ALL_FEATURES[feature["name"]](**feature["args"])
        self.backbone = ALL_BACKBONES[backbone["name"]](**backbone["args"])
        self.linear = nn.Linear(self.backbone.output_dim, LKE_NUM_CLASSES)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=LKE_NUM_CLASSES, average="micro", ignore_index=-1)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=LKE_NUM_CLASSES, average="micro", ignore_index=-1)
        self.test_metric = LKEMetric()

    def training_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        x, y = self.feature_layer(x, y, shift=True)

        x = self.backbone(x)
        y_pred = self.linear(x).transpose(-1, -2)

        loss = self.loss_fn(y_pred, y)

        self.train_acc.update(y_pred, y)
        self.log("loss/train", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        x, y = self.feature_layer(x, y, shift=False)

        x = self.backbone(x)
        y_pred = self.linear(x).transpose(-1, -2)

        loss = self.loss_fn(y_pred, y)

        self.val_acc.update(y_pred, y)
        self.log("loss/val", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch["x"], batch["y"]
        x, y = self.feature_layer(x, y, shift=False)

        x = self.backbone(x)
        y_pred = self.linear(x).transpose(-1, -2)

        loss = self.loss_fn(y_pred, y)

        self.test_metric.update(y_pred, y)
        self.log("loss/test", loss)
        return loss

    def predict_step(self, x):
        x, _ = self.feature_layer(x, shift=False)

        x = self.backbone(x)
        y_pred = self.linear(x).transpose(-1, -2)

        return x, y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def on_train_epoch_end(self) -> None:
        acc = self.train_acc.compute()
        self.log("recall/train", acc, prog_bar=True)
        self.train_acc.reset()

    def on_validation_epoch_end(self) -> None:
        lke_acc = self.val_acc.compute()
        self.log("recall/val", lke_acc, prog_bar=True)
        self.val_acc.reset()

    def on_test_epoch_end(self) -> None:
        lke_metric_dict = self.test_metric.compute()
        lke_metric_dict = {f"test/{k}": v for k, v in lke_metric_dict.items()}
        self.log_dict(lke_metric_dict)
        self.test_metric.reset()
