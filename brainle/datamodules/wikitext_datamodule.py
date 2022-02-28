from typing import Any, Callable, Optional

import pytorch_lightning as pl
import torch
import torchvision
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split

from .datasets.masked_char_dataset import MaskedCharDataset


class WikiTextDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        train_val_split: tuple,
        num_workers: int,
        block_size: int,
        p_word_mask: float,
        p_char_mask: float,
        version: str = "wikitext-2-raw-v1",
        pin_memory: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.num_workers = num_workers
        self.block_size = block_size
        self.p_word_mask = p_word_mask
        self.p_char_mask = p_char_mask
        self.version = version
        self.pin_memory = pin_memory
        self.dataset: Any = None
        self.dataset_train: Any = None
        self.dataset_valid: Any = None

    def get_wikitext(self, split):
        self.dataset = load_dataset("wikitext", self.version, split=split)
        text = ""
        for i in range(len(self.dataset)):
            text += self.dataset[i]["text"]
        return text

    def setup(self, stage: Any = None) -> None:
        self.dataset = MaskedCharDataset(
            text=self.get_wikitext("train"),
            block_size=self.block_size,
            p_word_mask=self.p_word_mask,
            p_char_mask=self.p_char_mask,
        )
        self.dataset_train, self.dataset_valid = random_split(
            self.dataset, self.train_val_split
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset_valid,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
