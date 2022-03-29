from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from .architectures.fovea_cropper import FoveaCropper


class MQFCModel(pl.LightningModule):

    """Vector Quantization and Fovea Cropper"""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        quantizer: nn.Module,
        crop_sizes: List[int],
        crop_res: int,
        learning_rate: float,
        loss_quantize_weight: float,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.crop_sizes = crop_sizes
        self.loss_quantize_weight = loss_quantize_weight

        self.cropper = FoveaCropper(out_size=crop_res, sizes=crop_sizes)
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer

    def forward(self, x: torch.Tensor):
        x_crops = self.cropper(x)
        z = self.encoder(x_crops)
        b, c, h, w = z.shape
        z = rearrange(z, "b c h w -> b (h w) c")
        quantize = self.quantizer(z)
        x_pred = self.decoder(rearrange(z, "b (h w) c -> b c h w", h=h, w=w))
        return x_crops, x_pred, z, quantize

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quantizer.parameters()),
            lr=self.learning_rate,
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        batch_crops, batch_pred, batch_z, quantize = self(batch)

        loss_recon = F.mse_loss(batch_crops, batch_pred)
        loss_quantize = F.mse_loss(quantize["embedding"].detach(), batch_z)

        loss = loss_recon + self.loss_quantize_weight * loss_quantize

        self.log("train_loss", loss)
        log_dict = {
            "train_loss_recon": loss_recon,
            "train_loss_quantize": loss_quantize,
        }
        log_dict.update(
            {
                f"train_perplexity_{i}": quantize["perplexity"][i]
                for i in range(len(quantize["perplexity"]))
            }
        )
        self.log_dict(log_dict)

        return loss

    def validation_step(self, batch, batch_idx):
        batch_crops, batch_pred, batch_z, quantize = self(batch)

        loss_recon = F.mse_loss(batch_crops, batch_pred)
        loss_quantize = F.mse_loss(quantize["embedding"].detach(), batch_z)

        loss = loss_recon + self.loss_quantize_weight * loss_quantize

        self.log("valid_loss", loss)
        log_dict = {
            "valid_loss_recon": loss_recon,
            "valid_loss_quantize": loss_quantize,
        }
        log_dict.update(
            {
                f"valid_perplexity_{i}": quantize["perplexity"][i]
                for i in range(len(quantize["perplexity"]))
            }
        )
        self.log_dict(log_dict)

        return loss
