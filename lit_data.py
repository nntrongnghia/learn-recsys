from abc import ABC, abstractmethod
from typing import Tuple
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

class BaseDataset(Dataset, ABC):
    @abstractmethod
    def split(self, *args, **kwargs) -> Tuple[Dataset, Dataset]:
        """Split the dataset into train split
        and test/validation split

        Returns
        -------
        Tuple[Dataset, Dataset]
            Two Dataset instances for training and validation/testing
        """

class LitDataModule(pl.LightningDataModule):
    def __init__(self, dataset: BaseDataset,
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
        self.num_users = getattr(self.dataset, "num_users", None)
        self.num_items = getattr(self.dataset, "num_items", None)
        self.train_split, self.test_split = self.dataset.split(
            self.train_ratio)

    def train_dataloader(self):
        return DataLoader(self.train_split, **self.dataloader_kwargs, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_split, **self.dataloader_kwargs, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_split, **self.dataloader_kwargs, shuffle=False)
