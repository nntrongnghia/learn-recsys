from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger

from lit_model import LitModel
from ml100k import LitDataModule, ML100KRatingMatrix


class AutoRec(nn.Module):
    def __init__(self, embedding_dims, input_dim, dropout=0.05):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embedding_dims),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        )
        self.decoder = nn.Linear(embedding_dims, input_dim)
        for param in self.parameters():
            nn.init.normal_(param, std=0.01)

    def forward(self, x):
        h = self.encoder(x)
        y = self.decoder(h)
        return y

class LitAutoRec(LitModel):
    def get_loss(self, m_outputs, batch):
        mask = (batch > 0).to(torch.float32)
        m_outputs = m_outputs*mask
        return F.mse_loss(m_outputs, batch) 

    def update_metric(self, m_outputs, batch):
        mask = batch > 0
        self.rmse.update(m_outputs[mask], batch[mask])

    def forward(self, batch):
        return self.model(batch)


def main(args):
    data = LitDataModule(ML100KRatingMatrix(), batch_size=args.batch_size)
    data.setup()
    model = LitAutoRec(AutoRec, 
            lr=0.01,
            input_dim=data.num_users,
            embedding_dims=args.embedding_dims)
    
    logger = TensorBoardLogger("lightning_logs", name=f"AutoRec_{args.embedding_dims}")
    trainer = pl.Trainer.from_argparse_args(args, logger=logger)
    trainer.fit(model, data)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--embedding_dims", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)


