from typing import Any, Callable, Optional

import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import DataLoader, random_split

from .datasets.images_archive_dataset import ImagesArchiveDataset


class ImageDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        train_val_split: tuple,
        batch_size: int,
        num_workers: int,
        transform: Callable = torchvision.transforms.ToTensor(),
        pin_memory: bool = False,
        **kwargs: int,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.train_val_split = train_val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.pin_memory = pin_memory
        self.data_train: Any = None
        self.data_val: Any = None

    def setup(self, stage: Any = None) -> None:
        # Transform and split datasets
        trainset = ImagesArchiveDataset(
            archive_dir=self.data_dir, transform=self.transform
        )
        self.data_train, self.data_val = random_split(trainset, self.train_val_split)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
