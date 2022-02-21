from argparse import ArgumentParser
from turtle import forward
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import RetrievalHitRate

from lit_data import LitDataModule
from lit_model import LitModel
from ml100k import ML100KPairWise
from utils import bpr_loss


class NeuMF(nn.Module):
    def __init__(self, embedding_dims: int, num_users: int, num_items: int, hidden_dims: List, **kwargs):
        super().__init__()
        self.P = nn.Embedding(num_users, embedding_dims)
        self.Q = nn.Embedding(num_items, embedding_dims)
        self.U = nn.Embedding(num_users, embedding_dims)
        self.V = nn.Embedding(num_items, embedding_dims)
        mlp = [nn.Linear(embedding_dims*2, hidden_dims[0]),
               nn.ReLU()]
        for i in range(len(hidden_dims) - 1):
            mlp += [nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.ReLU()]
        self.mlp = nn.Sequential(*mlp)
        self.output_layer = nn.Linear(
            hidden_dims[-1] + embedding_dims, 1, bias=False)

    def forward(self, user_id, item_id) -> torch.Tensor:
        p_mf = self.P(user_id)
        q_mf = self.Q(item_id)
        gmf = p_mf * q_mf

        p_mlp = self.U(user_id)
        q_mlp = self.V(item_id)
        mlp = self.mlp(torch.cat([p_mlp, q_mlp], axis=-1))
        logit = self.output_layer(
            torch.cat([gmf, mlp], axis=-1))
        return logit


class LitNeuMF(pl.LightningModule):
    def __init__(self, lr=0.002, hitrate_cutout=10, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = NeuMF(**kwargs)
        self.lr = lr
        self.hitrate = RetrievalHitRate(k=hitrate_cutout)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr, weight_decay=1e-5)

    def forward(self, user_id, item_id):
        return self.model(user_id, item_id)

    def training_step(self, batch, batch_idx):
        user_id, pos_item, neg_item = batch
        pos_score = self(user_id, pos_item)
        neg_score = self(user_id, neg_item)
        loss = bpr_loss(pos_score, neg_score)
        return loss

    def validation_step(self, batch, batch_idx):
        user_id, item_id, is_pos = batch
        logit = self(user_id, item_id)
        score = torch.sigmoid(logit).reshape(-1,)
        self.hitrate.update(score, is_pos, user_id)
        return

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.logger.experiment.add_scalar(
            "train/loss", avg_loss, self.current_epoch)

    def validation_epoch_end(self, outputs):
        self.logger.experiment.add_scalar(
            f"val/hit_rate@{self.hitrate.k}",
            self.hitrate.compute(),
            self.current_epoch)
        self.hitrate.reset()


def main(args):
    data = LitDataModule(
        ML100KPairWise(test_sample_size=100),
        batch_size=args.batch_size)
    data.setup()
    model = LitNeuMF(
        num_users=data.num_users, num_items=data.num_items,
        embedding_dims=args.embedding_dims,
        hidden_dims=[10, 10, 10]
    )

    logger = TensorBoardLogger("lightning_logs", name=f"NeuMF")
    trainer = pl.Trainer.from_argparse_args(args, logger=logger)

    trainer.fit(model, data)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--embedding_dims", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1024)
    pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
