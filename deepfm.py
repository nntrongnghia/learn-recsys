from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import Accuracy
from fm import LitFM

from lit_data import LitDataModule
from lit_model import LitModel
from ctr import CTRDataset


def mlp_layer(in_dim, out_dim, dropout=0.0):
    return [
        nn.Linear(in_dim, out_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
    ]


class DeepFM(nn.Module):
    def __init__(self, feat_dims, embedding_dims,
                 mlp_dims=[30, 20, 10], dropout=0.1):
        super().__init__()
        num_inputs = int(sum(feat_dims))
        self.embed_output_dim = len(feat_dims) * embedding_dims
        self.embedding = nn.Embedding(num_inputs, embedding_dims)
        self.proj = nn.Embedding(num_inputs, 1)
        self.fc = nn.Linear(1, 1)
        self.mlp = nn.Sequential(
            *mlp_layer(self.embed_output_dim, mlp_dims[0], dropout),
            *[layer for i in range(len(mlp_dims) - 1)
              for layer in mlp_layer(mlp_dims[i], mlp_dims[i+1], dropout)],
            nn.Linear(mlp_dims[-1], 1))
        self.init_param()

    def init_param(self):
        for param in self.parameters():
            try:
                nn.init.xavier_normal_(param)
            finally:
                continue

    def forward(self, x):
        v = self.embedding(x)
        # Factorization Machine
        fm_interaction = 1/2*(v.sum(1)**2 - (v**2).sum(1)
                              ).sum(-1, keepdims=True)
        fm_proj = self.proj(x).sum(1)
        fm_logit = self.fc(fm_proj + fm_interaction).flatten()
        # MLP
        mlp_logit = self.mlp(v.flatten(1)).flatten()
        logit = fm_logit + mlp_logit
        return torch.sigmoid(logit)


class LitDeepFM(LitFM):
    def __init__(self, lr=0.002, **kwargs):
        super(LitFM, self).__init__()
        self.save_hyperparameters()
        self.model = DeepFM(**kwargs)
        self.lr = lr
        self.train_acc = Accuracy()
        self.test_acc = Accuracy()


def main(args):
    data = LitDataModule(
        CTRDataset(), batch_size=args.batch_size)
    data.setup()

    model = LitDeepFM(
        feat_dims=data.dataset.feat_dims,
        embedding_dims=args.embedding_dims)

    logger = TensorBoardLogger(
        "lightning_logs", name=f"DeepFM_{args.embedding_dims}")
    trainer = pl.Trainer.from_argparse_args(args, logger=logger)
    trainer.fit(model, data)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--embedding_dims", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1024)
    pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
