from copy import deepcopy
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split


def read_data_ml100k(data_dir="./ml-100k") -> pd.DataFrame:
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(os.path.join(data_dir, 'u.data'), '\t', names=names)
    return data


class ML100K(Dataset):
    def __init__(self, data_dir="./ml-100k", normalize_rating=False):
        self.normalize_rating = normalize_rating
        self.data_dir = data_dir
        self.df = read_data_ml100k(data_dir)
        # set to zero-based index
        self.df.user_id -= 1
        self.df.item_id -= 1
        if normalize_rating:
            self.df.rating /= 5.0
        self.num_users = self.df.user_id.unique().shape[0]
        self.num_items = self.df.item_id.unique().shape[0]
        self.user_id = self.df.user_id.values
        self.item_id = self.df.item_id.values
        self.rating = self.df.rating.values.astype(np.float32)
        self.timestamp = self.df.timestamp

    def split(self, train_ratio=0.8):
        train_len = int(train_ratio*len(self))
        test_len = len(self) - train_len
        return random_split(self, [train_len, test_len])
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.user_id[idx], self.item_id[idx], self.rating[idx]


class ML100KRatingMatrix(ML100K):
    def __init__(self, data_dir="./ml-100k", user_based=False, normalize_rating=False):
        super().__init__(data_dir)
        self.normalize_rating = normalize_rating
        self.user_based = user_based
        self.rating_matrix = np.zeros(
            (self.num_items, self.num_users), dtype=np.float32)
        self.rating_matrix[[self.item_id, self.user_id]] = self.rating
        if normalize_rating:
            self.rating_matrix /= 5.0

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


class ML100KPairWise(ML100K):
    def __init__(self, data_dir="./ml-100k", 
                 test_leave_out=1,
                 seq_test_sample_size=None):
        super().__init__(data_dir)
        self.set_all_item_ids = set(np.unique(self.item_id))
        # for training
        self.observed_items_per_user = None
        self.unobserved_items_per_user = None
        # for testing
        self.test_leave_out = test_leave_out
        self.seq_test_sample_size = seq_test_sample_size
        self.gt_pos_items_per_user = None
        # general
        self.train = None
        self.has_setup = False

    def split(self, *args, **kwargs):
        user_group = self.df.groupby("user_id", sort=False)
        train_df = []
        test_df = []
        for user_id, user_df in user_group:
            user_df = user_df.sort_values("timestamp")
            train_df.append(user_df[:-self.test_leave_out])
            test_df.append(user_df[-self.test_leave_out:])
        train_df = pd.concat(train_df)
        test_df = pd.concat(test_df)
        # Train split
        train_split = deepcopy(self)
        train_split.user_id = train_df.user_id.values
        train_split.item_id = train_df.item_id.values
        train_split.observed_items_per_user = {
            int(user_id): user_df.item_id.values
            for user_id, user_df in train_df.groupby("user_id", sort=False)
        }
        train_split.unobserved_items_per_user = {
            user_id: np.array(
                list(self.set_all_item_ids - set(observed_items)))
            for user_id, observed_items in train_split.observed_items_per_user.items()
        }
        train_split.train = True
        train_split.has_setup = True
        # Test split
        test_split = deepcopy(self)
        test_split.user_id = []
        test_split.item_id = []
        test_split.gt_pos_items_per_user = {
            int(user_id): user_df.item_id.values
            for user_id, user_df in test_df.groupby("user_id", sort=False)
        }
        for user_id, items in train_split.unobserved_items_per_user.items():
            if self.seq_test_sample_size is None:
                sample_items = items
            elif isinstance(self.seq_test_sample_size, int):
                sample_items = np.random.choice(items, self.seq_test_sample_size)
            else:
                raise TypeError("self.seq_test_sample_size should be int")
            sample_items = np.concatenate(
                [test_split.gt_pos_items_per_user[user_id],
                 sample_items])
            sample_items = np.unique(sample_items)
            test_split.user_id += [user_id]*len(sample_items)
            test_split.item_id.append(sample_items)
        test_split.user_id = np.array(test_split.user_id)
        test_split.item_id = np.concatenate(test_split.item_id)
        test_split.train = False
        test_split.has_setup = True

        return train_split, test_split

    def __len__(self):
        return len(self.user_id)

    def __getitem__(self, idx):
        assert self.has_setup, "Must run self.setup()"
        if self.train:
            user_id = self.user_id[idx]
            pos_item = self.item_id[idx]
            neg_item = np.random.choice(
                self.unobserved_items_per_user[int(user_id)])
            return user_id, pos_item, neg_item
        else:
            user_id = self.user_id[idx]
            item_id = self.item_id[idx]
            is_pos = item_id in self.gt_pos_items_per_user[user_id]
            return user_id, item_id, is_pos


class LitDataModule(pl.LightningDataModule):
    def __init__(self, dataset: ML100K,
                 train_ratio=0.8, batch_size=32,
                 num_workers=2, prefetch_factor=16):
        self.dataset = dataset
        self.train_ratio = train_ratio
        self.dataloader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "prefetch_factor": prefetch_factor,
        }

    def setup(self):
        self.num_users = self.dataset.num_users
        self.num_items = self.dataset.num_items
        self.train_split, self.test_split = self.dataset.split(self.train_ratio)

    def train_dataloader(self):
        return DataLoader(self.train_split, **self.dataloader_kwargs, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_split, **self.dataloader_kwargs, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_split, **self.dataloader_kwargs, shuffle=False)


if __name__ == "__main__":
    import d2l.mxnet as d2l
    dataset = ML100K()
    dm = LitDataModule(dataset)
    dm.setup()

    # df = read_data_ml100k()
    # num_users = df.user_id.unique().shape[0]
    # num_items = df.item_id.unique().shape[0]
    # train_data, test_data = d2l.split_data_ml100k(df, num_users, num_items,
    #                                               'seq-aware')
    # users_train, items_train, ratings_train, candidates = d2l.load_data_ml100k(
    #     train_data, num_users, num_items, feedback="implicit")
    # users_test, items_test, ratings_test, test_iter = d2l.load_data_ml100k(
    #     test_data, num_users, num_items, feedback="implicit")
    print("HOLD")
