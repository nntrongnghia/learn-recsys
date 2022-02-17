from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger

from lit_model import LitModel
from ml100k import LitML100K, ML100KRatingMatrix


class AutoRec(nn.Module):
    def __init__(self, num_features, input_dim, dropout=0.05):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, num_features),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        )
        self.decoder = nn.Linear(num_features, input_dim)
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
    data = LitML100K(batch_size=args.batch_size, return_rating_matrix=True)
    data.setup()
    model = LitAutoRec(AutoRec, 
            lr=0.01,
            input_dim=data.num_users,
            num_features=args.num_features)
    
    logger = TensorBoardLogger("lightning_logs", name=f"AutoRec_{args.num_features}")
    trainer = pl.Trainer.from_argparse_args(args, logger=logger)
    trainer.fit(model, data)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_features", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)


