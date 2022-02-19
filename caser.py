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
from ml100k import LitDataModule
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


# for code dev
# if __name__ == "__main__":
#     model = Caser(28, 15, 20)
#     user_id = torch.tensor([10, 0, 4, 2])
#     seq = torch.tensor([
#         [0, 0, 2, 1, 0],
#         [1, 2, 3, 6, 0],
#         [0, 8, 2, 1, 7],
#         [9, 3, 5, 3, 0]
#     ])
#     item_id = torch.tensor([0, 1, 2, 3])
#     model(user_id, seq, item_id)
