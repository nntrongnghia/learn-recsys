from argparse import ArgumentParser
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics import MeanSquaredError


def read_data_ml100k(data_dir="./ml-100k") -> pd.DataFrame:
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(os.path.join(data_dir, 'u.data'), '\t', names=names)
    return data


class ML100K(Dataset):
    def __init__(self, data_dir="./ml-100k"):
        self.data_dir = data_dir
        self.df = read_data_ml100k(data_dir)
        self.num_users = self.df.user_id.unique().shape[0]
        self.num_items = self.df.item_id.unique().shape[0]
        self.user_id = self.df.user_id.values - 1
        self.item_id = self.df.item_id.values - 1
        self.rating = self.df.rating.values.astype(np.float32)
        

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.user_id[idx], self.item_id[idx], self.rating[idx]


class ML100KRatingMatrix(ML100K):
    def __init__(self, data_dir="./ml-100k", user_based=False):
        super().__init__(data_dir)
        self.user_based = user_based
        self.rating_matrix = np.zeros((self.num_items, self.num_users), dtype=np.float32)
        self.rating_matrix[[self.item_id, self.user_id]] = self.rating

    def __len__(self):
        if self.user_based:
            return self.num_users
        else:
            return self.num_items

    def __getitem__(self, idx):
        if self.user_based:
            return self.rating_matrix[:, idx]
        else:
            return self.rating_matrix[idx]


class LitML100K(pl.LightningDataModule):
    def __init__(self, data_dir="./ml-100k",
                 return_rating_matrix=False,
                 train_ratio=0.8, batch_size=32,
                 num_workers=2, prefetch_factor=16):
        self.return_rating_matrix = return_rating_matrix
        self.data_dir = data_dir
        self.train_ratio = train_ratio
        self.dataloader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "prefetch_factor": prefetch_factor,
        }

    def setup(self):
        if self.return_rating_matrix:
            self.data = ML100KRatingMatrix(self.data_dir)
        else:
            self.data = ML100K(self.data_dir)
        self.num_users = self.data.num_users
        self.num_items = self.data.num_items
        self.train_len = int(self.train_ratio*len(self.data))
        self.test_len = len(self.data) - self.train_len
        self.train_split, self.test_split = random_split(
            self.data, [self.train_len, self.test_len])

    def train_dataloader(self):
        return DataLoader(self.train_split, **self.dataloader_kwargs, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_split, **self.dataloader_kwargs, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_split, **self.dataloader_kwargs, shuffle=False)


if __name__ == "__main__":
    data = ML100KRatingMatrix()
    print("HOLD")