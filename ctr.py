import os
from collections import defaultdict
from copy import deepcopy
from typing import Tuple

import numpy as np
import pandas as pd

from lit_data import BaseDataset


def csv_reader(data_path):
    dtype = {i: str for i in range(1, 35)}
    dtype[0] = np.uint8
    return pd.read_csv(data_path, "\t",
                       header=None, dtype=dtype)


class CTRDataset(BaseDataset):
    def __init__(self, data_dir="./ctr", min_threshold=4):
        """Read CTR dataset from train.csv and test.csv

        Parameters
        ----------
        data_dir : str
            Path to directory containing train.csv and test.csv
        min_threshold : int, optional
            Remove feature values that occurs less than this threshold
            By default 4
        """
        # Read csv
        self.data_dir = data_dir
        self.train_df = csv_reader(os.path.join(data_dir, "train.csv"))
        self.test_df = csv_reader(os.path.join(data_dir, "test.csv"))
        self.feat_cols = [i for i in range(1, len(self.train_df.columns))]
        # Count unique values in each columns
        feat_counts = {
            col: self.train_df[col].value_counts()
            for col in self.feat_cols}
        # Feature mapper maps a unique encoded value to an identifier
        # So each value is considered to be a categorical value.
        # Unique values are filtered with occurence greater or equal to min_threshold
        # A default value will be assign to values that not defined in feature mapper
        self.feat_mapper = {}

        def _constant_factory(v):
            return lambda: v
        for col, val_counts in feat_counts.items():
            val = val_counts.index[val_counts >= min_threshold]
            default = len(val)
            self.feat_mapper[col] = pd.Series(
                range(len(val)), index=val, dtype=np.int32
            ).to_dict(defaultdict(_constant_factory(default)))
        # Feature dimension = number of unique values = number of values in mapper + defaults
        self.feat_dims = np.array([len(mapper) + 1
                                   for mapper in self.feat_mapper.values()])
        # Offset is a value add to the whole field to discriminate values in different columns
        self.offsets = np.array((0, *np.cumsum(self.feat_dims).tolist()[:-1])).astype(np.int32)
        # Map values in dataframe
        for col, mapper in self.feat_mapper.items():
            self.train_df[col] = self.train_df[col].map(mapper)
            self.test_df[col] = self.test_df[col].map(mapper)
        # For each split
        self.X = None
        self.y = None

    def build_items(self, train=True):
        if train:
            df = self.train_df
        else:
            df = self.test_df
        self.X = df[self.feat_cols].values + self.offsets
        self.y = df[0].values

    def split(self, *args, **kwargs) -> Tuple[BaseDataset, BaseDataset]:
        train_split = deepcopy(self)
        train_split.build_items(True)

        test_split = deepcopy(self)
        test_split.build_items(False)

        return train_split, test_split

    def __len__(self):
        assert self.X is not None and self.y is not None
        return len(self.X)

    def __getitem__(self, idx):
        assert self.X is not None and self.y is not None
        return self.X[idx], self.y[idx]


if __name__ == "__main__":
    data = CTRDataset("ctr")
    train_split, test_split = data.split()
    print(train_split[0])
