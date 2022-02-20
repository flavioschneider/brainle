from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from .architectures.convnext import ConvNeXtDecoder, ConvNeXtEncoder
from .architectures.fovea_cropper import FoveaCropper
from .architectures.quantizer import VectorQuantizer


class VQVAEModel(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        embedding_dim: int,
        num_embeddings: int,
        learning_rate: float,
        beta: float,
    ):
        super().__init__()
        self.learning_rate = learning_rate

        self.encoder = ConvNeXtEncoder(
            in_channels=in_channels, depths=[3, 3], dims=[96, embedding_dim]
        )
        self.decoder = ConvNeXtDecoder(
            in_channels=embedding_dim, depths=[3, 3, 3], dims=[96, 24, in_channels]
        )
        self.batchnorm = nn.BatchNorm2d(num_features=embedding_dim)
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim, beta=beta
        )

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        quantize = self.quantizer(self.batchnorm(z))
        x_pred = self.decoder(quantize["quantized"])
        return x_pred, quantize

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quantizer.parameters()),
            lr=self.learning_rate,
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        batch_pred, quantize = self(batch)

        loss_recon = F.mse_loss(batch, batch_pred)
        loss_quantize = quantize["loss"]
        loss = loss_recon + loss_quantize

        self.log("train_loss", loss)
        self.log_dict(
            {
                "train_perplexity": quantize["perplexity"],
                "train_loss_recon": loss_recon,
                "train_loss_quantize": loss_quantize,
            }
        )
        return loss

    def validation_step(self, batch, batch_idx):
        batch_pred, quantize = self(batch)

        loss_recon = F.mse_loss(batch, batch_pred)
        loss_quantize = quantize["loss"]
        loss = loss_recon + loss_quantize

        self.log("valid_loss", loss)
        self.log_dict(
            {
                "valid_perplexity": quantize["perplexity"],
                "valid_loss_recon": loss_recon,
                "valid_loss_quantize": loss_quantize,
            }
        )

        return loss


class VQVAEFCModel(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        embedding_dim: int,
        num_embeddings: int,
        crop_sizes: List[int],
        crop_res: int,
        learning_rate: float,
        beta: float,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.crop_sizes = crop_sizes

        self.cropper = FoveaCropper(out_size=crop_res, sizes=crop_sizes)
        self.encoder = ConvNeXtEncoder(
            in_channels=in_channels, depths=[6], dims=[embedding_dim]
        )
        self.decoder = ConvNeXtDecoder(
            in_channels=embedding_dim, depths=[3, 6], dims=[32, in_channels]
        )
        self.batchnorm = nn.BatchNorm2d(num_features=embedding_dim)
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim, beta=beta
        )

    def forward(self, x: torch.Tensor):
        x_crops = self.cropper(x)
        z = self.encoder(x_crops)
        quantize = self.quantizer(self.batchnorm(z))
        x_pred = self.decoder(quantize["quantized"])
        return x_crops, x_pred, quantize

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quantizer.parameters()),
            lr=self.learning_rate,
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        batch_crops, batch_pred, quantize = self(batch)

        loss_recon = F.mse_loss(batch_crops, batch_pred)
        loss_quantize = quantize["loss"]
        loss = loss_recon + loss_quantize

        self.log("train_loss", loss)
        self.log_dict(
            {
                "train_perplexity": quantize["perplexity"],
                "train_loss_recon": loss_recon,
                "train_loss_quantize": loss_quantize,
            }
        )
        return loss

    def validation_step(self, batch, batch_idx):
        batch_crops, batch_pred, quantize = self(batch)

        loss_recon = F.mse_loss(batch_crops, batch_pred)
        loss_quantize = quantize["loss"]
        loss = loss_recon + loss_quantize

        self.log("valid_loss", loss)
        self.log_dict(
            {
                "valid_perplexity": quantize["perplexity"],
                "valid_loss_recon": loss_recon,
                "valid_loss_quantize": loss_quantize,
            }
        )

        return loss
