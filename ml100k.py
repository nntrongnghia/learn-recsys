from abc import ABC, abstractmethod
from copy import deepcopy
import os
from typing import Tuple
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch import unbind
from torch.utils.data import DataLoader, Dataset, random_split


def read_data_ml100k(data_dir="./ml-100k") -> pd.DataFrame:
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(os.path.join(data_dir, 'u.data'), '\t', names=names)
    return data


class BaseML100K(Dataset, ABC):
    @abstractmethod
    def split(self, *args, **kwargs) -> Tuple[Dataset, Dataset]:
        """Split the dataset into train split
        and test/validation split

        Returns
        -------
        Tuple[Dataset, Dataset]
            Two Dataset instances for training and validation/testing
        """


class ML100K(BaseML100K):
    def __init__(self, data_dir="./ml-100k", normalize_rating=False):
        """MovieLens 100K for Matrix Factorization
        Each sample is a tuple of:
        - user_id: int
        - item_id: int
        - rating: float

        Parameters
        ----------
        data_dir : str, optional
            Path to dataset directory, by default "./ml-100k"
        normalize_rating : bool, optional
            If True, rating is normalized to (0..1), by default False
        """
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
        """MovieLens 100K for AutoRec
        Each sample is a row/column of the rating matrix.
        The rating matrix has shape (number of items, number of users).

        Parameters
        ----------
        data_dir : str, optional
            _description_, by default "./ml-100k"
        user_based : bool, optional
            If True, extract columns of the rating matrix.
            else, extract rows.
            By default False
        normalize_rating : bool, optional
            If True, divide rating by 5, by default False
        """
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
                 test_sample_size:int=None):
        """Pair Wise loader to train NeuMF model.
        Samples are slightly different based on train/test mode.
        
        In training mode:
        - user_id: int
        - item_id: int
            Item id that user has interacted with
        - neg_item: int
            Item id that user hasn't interacted with while training
        
        In testing mode:
        - user_id: int
        - item_id: int
            Random item_id to be ranked by the model
        - is_pos: bool
            If True, this item is a positive item 
            that user has interacted with in groundtruth data.


        Parameters
        ----------
        data_dir : str, optional
            Path to dataset directory, by default "./ml-100k"
        test_leave_out : int, optional
            Leave out how many items per user for testing
            By default 1
        test_sample_size : int, optional
            It is time-consuming to rank all items for every user during
            evaluation, we can randomly choose a subset of items to rank
            If None, rank all items.
            By default None
        """
        super().__init__(data_dir)
        self.set_all_item_ids = set(np.unique(self.item_id))
        self.test_leave_out = test_leave_out
        self.test_sample_size = test_sample_size
        # general
        self.train = None
        self.has_setup = False
        # Split Dataframe
        self.split_dataframe()
        self.build_candidates()

    def split_dataframe(self):
        """Split ML100K dataframe with the strategy leave-n-out
        with timestamp order.
        """
        user_group = self.df.groupby("user_id", sort=False)
        train_df = []
        test_df = []
        for user_id, user_df in user_group:
            user_df = user_df.sort_values("timestamp")
            train_df.append(user_df[:-self.test_leave_out])
            test_df.append(user_df[-self.test_leave_out:])
        self.train_df = pd.concat(train_df)
        self.test_df = pd.concat(test_df)

    def build_candidates(self):
        # Train
        self.observed_items_per_user_in_train = {
            int(user_id): user_df.item_id.values
            for user_id, user_df in self.train_df.groupby("user_id", sort=False)
        }
        self.unobserved_items_per_user_in_train = {
            user_id: np.array(
                list(self.set_all_item_ids - set(observed_items)))
            for user_id, observed_items in self.observed_items_per_user_in_train.items()
        }
        # Test
        self.gt_pos_items_per_user_in_test = {
            int(user_id): user_df[-self.test_leave_out:].item_id.values
            for user_id, user_df in self.test_df.groupby("user_id", sort=False)
        }

    def split(self, *args, **kwargs):
        # Train split
        train_split = deepcopy(self)
        train_split.user_id = self.train_df.user_id.values
        train_split.item_id = self.train_df.item_id.values
        train_split.train = True
        train_split.has_setup = True
        # Test split
        test_split = deepcopy(self)
        test_split.user_id = []
        test_split.item_id = []
        for user_id, items in self.unobserved_items_per_user_in_train.items():
            if self.test_sample_size is None:
                sample_items = items
            elif isinstance(self.test_sample_size, int):
                sample_items = np.random.choice(items, self.test_sample_size)
            else:
                raise TypeError("self.test_sample_size should be int")
            sample_items = np.concatenate(
                [test_split.gt_pos_items_per_user_in_test[user_id],
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
                self.unobserved_items_per_user_in_train[int(user_id)])
            return user_id, pos_item, neg_item
        else:
            user_id = self.user_id[idx]
            item_id = self.item_id[idx]
            is_pos = item_id in self.gt_pos_items_per_user_in_test[user_id]
            return user_id, item_id, is_pos


class ML100KSequence(ML100KPairWise):
    def __init__(self, data_dir="./ml-100k",
                 test_leave_out=1,
                 test_sample_size=100,
                 seq_len=5):
        """Sequence data to train Caser model
        Similarly to Pair Wise dataset, the sample depends on train/test mode.

        In training mode:
        - user_id: int
        - seq: List[int]
            Sequence of last N item ids that user has interacted with.
        - target_item: int
            Target item id that user will interact with after the sequence
        - neg_item: int
            Item id that user doesn't interacted with while training
        
        In testing mode:
        - user_id: int
        - seq: List[int]
            Sequence of last N item ids that user has interacted with.
        - item_id: int
            Random item_id to be ranked by the model
        - is_pos: bool
            If True, this item is a positive item 
            that user has interacted with in groundtruth data.

        Parameters
        ----------
        data_dir : str, optional
            Path to dataset directory, by default "./ml-100k"
        test_leave_out : int, optional
            Leave out how many items per user for testing
            By default 1
        test_sample_size : int, optional
            It is time-consuming to rank all items for every user during
            evaluation, we can randomly choose a subset of items to rank
            If None, rank all items.
            By default None
        seq_len : int, optional
            Length of sequence of item ids, by default 5
        """
        self.seq_len = seq_len
        super().__init__(data_dir, test_leave_out, test_sample_size)
        self.getitem_df = None

    def split_dataframe(self):
        user_group = self.df.groupby("user_id", sort=False)
        train_df = []
        test_df = []
        for user_id, user_df in user_group:
            user_df = user_df.sort_values("timestamp")
            train_df.append(user_df[:-self.test_leave_out])
            test_df.append(user_df[-(self.test_leave_out+self.seq_len):])
        self.train_df = pd.concat(train_df)
        self.test_df = pd.concat(test_df)

    def split(self, *args, **kwargs):
        # Train
        train_split = deepcopy(self)
        df = []
        for _, user_df in self.train_df.groupby("user_id", sort=False):
            user_df = user_df.sort_values("timestamp").reset_index()
            user_id = user_df.user_id[:-self.seq_len].values
            target = user_df.item_id[self.seq_len:].values
            seq = [
                user_df.item_id[i:i+self.seq_len].values
                for i in range(len(user_df) - self.seq_len)]
            df.append(
                pd.DataFrame({
                    "user_id": user_id,
                    "seq": seq,
                    "target_item": target}))
        train_split.getitem_df = pd.concat(df).reset_index()
        train_split.train = True

        # Test
        test_split = deepcopy(self)
        df = []
        for uid, user_df in self.test_df.groupby("user_id", sort=False):
            user_df = user_df.sort_values("timestamp").reset_index()
            user_id = user_df.user_id[:-self.seq_len].values
            seq = [
                user_df.item_id[i:i+self.seq_len].values
                for i in range(len(user_df) - self.seq_len)]
            target_per_seq = user_df.item_id[self.seq_len:].values
            unobserved_item_id = np.concatenate([
                np.random.choice(
                    self.unobserved_items_per_user_in_train[uid],
                    self.test_sample_size,
                    replace=self.test_sample_size > self.unobserved_items_per_user_in_train[uid].shape[0]),
            ])
            item_id_per_seq = [
                np.unique(np.append(unobserved_item_id, target))
                for target in target_per_seq
            ]
            user_id = np.concatenate([
                np.repeat(u, len(item_id))
                for u, item_id in zip(user_id, item_id_per_seq)
            ])
            seq = np.concatenate([
                np.repeat(s.reshape(1, -1), len(item_id), 0)
                for s, item_id in zip(seq, item_id_per_seq)
            ])
            item_id = np.concatenate(item_id_per_seq)
            is_pos = np.isin(item_id, target_per_seq)
            df.append(
                pd.DataFrame({
                    "user_id": user_id,
                    "seq": list(seq),
                    "item_id": item_id,
                    "is_pos": is_pos}))
        test_split.getitem_df = pd.concat(df).reset_index()
        test_split.train = False
        return train_split, test_split

    def __len__(self):
        assert self.getitem_df is not None
        return len(self.getitem_df)

    def __getitem__(self, idx):
        assert self.getitem_df is not None
        row = self.getitem_df.iloc[idx]
        if self.train:
            neg_item = np.random.choice(
                self.unobserved_items_per_user_in_train[int(row.user_id)])
            return row.user_id, row.seq, row.target_item, neg_item
        else:
            return row.user_id, row.seq, row.item_id, row.is_pos


class LitDataModule(pl.LightningDataModule):
    def __init__(self, dataset: BaseML100K,
                 train_ratio=0.8, batch_size=32,
                 num_workers=2, prefetch_factor=16):
        """DataModule for PyTorch Lightning

        Parameters
        ----------
        dataset : BaseML100K
        train_ratio : float, optional
            By default 0.8
        batch_size : int, optional
            By default 32
        num_workers : int, optional
            Number of multi-CPU to fetch data
            By default 2
        prefetch_factor : int, optional
            Number of batches to prefecth, by default 16
        """
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
        self.train_split, self.test_split = self.dataset.split(
            self.train_ratio)

    def train_dataloader(self):
        return DataLoader(self.train_split, **self.dataloader_kwargs, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_split, **self.dataloader_kwargs, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_split, **self.dataloader_kwargs, shuffle=False)
