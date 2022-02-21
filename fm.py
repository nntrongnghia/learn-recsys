from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import Accuracy

from lit_data import LitDataModule
from lit_model import LitModel
from ctr import CTRDataset


class FactorizationMachine(nn.Module):
    def __init__(self, feat_dims, embedding_dims):
        super().__init__()
        num_inputs = int(sum(feat_dims))
        self.embedding = nn.Embedding(num_inputs, embedding_dims)
        self.proj = nn.Embedding(num_inputs, 1)
        self.fc = nn.Linear(1, 1)
        for param in self.parameters():
            try:
                nn.init.xavier_normal_(param)
            finally:
                continue

    def forward(self, x):
        v = self.embedding(x)
        interaction = 1/2*(v.sum(1)**2 - (v**2).sum(1)).sum(-1, keepdims=True)
        proj = self.proj(x).sum(1)
        logit = self.fc(proj + interaction)
        return torch.sigmoid(logit).flatten()


class LitFM(pl.LightningModule):
    def __init__(self, lr=0.002, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = FactorizationMachine(**kwargs)
        self.lr = lr
        self.train_acc = Accuracy()
        self.test_acc = Accuracy()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr, weight_decay=1e-5)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        ypred = self(x)
        loss = F.binary_cross_entropy(ypred, y.to(torch.float32))
        self.train_acc.update(ypred, y)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        ypred = self(x)
        loss = F.binary_cross_entropy(ypred, y.to(torch.float32))
        self.test_acc.update(ypred, y)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        acc = self.train_acc.compute()
        self.train_acc.reset()
        self.logger.experiment.add_scalar(
            "train/loss", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar(
            "train/acc", acc, self.current_epoch)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        acc = self.test_acc.compute()
        self.test_acc.reset()
        self.logger.experiment.add_scalar(
            "val/loss", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar(
            "val/acc", acc, self.current_epoch)


def main(args):
    data = LitDataModule(
        CTRDataset(), 
        batch_size=args.batch_size,
        num_workers=3,
        prefetch_factor=4)
    data.setup()

    model = LitFM(
        feat_dims=data.dataset.feat_dims,
        embedding_dims=args.embedding_dims)

    logger = TensorBoardLogger("lightning_logs", name=f"FM_{args.embedding_dims}")
    trainer = pl.Trainer.from_argparse_args(args, logger=logger)
    trainer.fit(model, data)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--embedding_dims", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1024)
    pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
