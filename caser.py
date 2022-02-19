from argparse import ArgumentParser
from turtle import forward
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import RetrievalHitRate
from pytorch_lightning.loggers import TensorBoardLogger

from lit_model import LitModel
from ml100k import LitDataModule, ML100KSequence
from utils import bpr_loss


class Caser(nn.Module):
    def __init__(self, embedding_dims, num_users, num_items,
                 L=5, num_hfilters=16, num_vfilters=4,
                 dropout=0.05, **kwargs):
        super().__init__()
        self.P = nn.Embedding(num_users, embedding_dims)
        self.Q = nn.Embedding(num_items, embedding_dims)

        self.num_hfilters = num_hfilters
        self.num_vfilters = num_vfilters
        # Vertical convolution
        self.conv_v = nn.Conv2d(1, num_vfilters, (L, 1))
        # Horizontal convolutions
        self.conv_h = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, num_hfilters, (h, embedding_dims)),
                nn.ReLU(),
                nn.AdaptiveMaxPool2d((1, 1)))
            for h in range(1, L+1)])
        # Fully-connected layer
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(
                num_vfilters*embedding_dims + num_hfilters*L,
                embedding_dims),
            nn.ReLU())
        self.Q_out = nn.Embedding(num_items, 2*embedding_dims)
        self.b_out = nn.Embedding(num_items, 1)

    def forward(self, user_id, seq, item_id):
        item_emb = self.Q(seq).unsqueeze(1)
        user_emb = self.P(user_id)

        v = self.conv_v(item_emb)
        h = torch.cat([filt(item_emb) for filt in self.conv_h], axis=-2)
        x = self.fc(torch.cat([v.flatten(1), h.flatten(1)], -1))
        x = torch.cat([x, user_emb], -1)
        logit = (self.Q_out(item_id)*x).sum(-1) + self.b_out(item_id).squeeze()
        return logit


class LitCaser(pl.LightningModule):
    def __init__(self, lr=0.002, hitrate_cutout=10, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = Caser(**kwargs)
        self.lr = lr
        self.hitrate = RetrievalHitRate(k=hitrate_cutout)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr, weight_decay=1e-5)

    def forward(self, user_id, seq, item_id):
        return self.model(user_id, seq, item_id)

    def training_step(self, batch, batch_idx):
        user_id, seq, pos_item, neg_item = batch
        pos_logit = self(user_id, seq, pos_item)
        neg_logit = self(user_id, seq, neg_item)
        loss = bpr_loss(pos_logit, neg_logit)
        return loss

    def validation_step(self, batch, batch_idx):
        user_id, seq, item_id, is_pos = batch
        logit = self(user_id, seq, item_id)
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
        ML100KSequence(seq_len=args.seq_len),
        batch_size=args.batch_size)
    data.setup()
    model = LitCaser(
        num_users=data.num_users, num_items=data.num_items,
        embedding_dims=args.embedding_dims,
        seq_len=args.seq_len)

    logger = TensorBoardLogger("lightning_logs",
                               name=f"Caser_{args.embedding_dims}_L{args.seq_len}")
    trainer = pl.Trainer.from_argparse_args(args, logger=logger)
    trainer.fit(model, data)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--embedding_dims", type=int, default=10)
    parser.add_argument("--seq_len", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1024)
    pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
