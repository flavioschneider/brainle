from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .architectures.attention import SMUNet


class SMModel(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = model

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_masked = torch.clone(x)
        x_masked = x * mask
        return self.model(x_masked)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            list(self.model.parameters()),
            lr=self.learning_rate,
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        sequences, masks = batch
        sequences_pred = self(sequences, masks)

        loss = F.cross_entropy(
            rearrange(sequences_pred, "b s k -> (b s) k"),
            rearrange(sequences, "b s -> (b s)"),
        )

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        sequences, masks = batch
        sequences_pred = self(sequences, masks)

        loss = F.cross_entropy(
            rearrange(sequences_pred, "b s k -> (b s) k"),
            rearrange(sequences, "b s -> (b s)"),
        )

        self.log("vaild_loss", loss)
        return loss
