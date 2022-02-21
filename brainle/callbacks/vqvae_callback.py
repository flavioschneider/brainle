import numpy as np
import torch
import torchvision
import wandb
from einops import rearrange
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class VQVAEReconstructionLogger(Callback):
    def __init__(self, batch_frequency: int = 10, num_images: int = 10) -> None:
        self.batch_frequency = batch_frequency
        self.num_images = num_images

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        self.log_img(pl_module, batch, batch_idx, split="val")

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if batch_idx % self.batch_frequency == 0:

            is_train = pl_module.training

            if is_train:
                pl_module.eval()

            images = batch[0 : self.num_images]
            images_reconstructed, quantize = pl_module(images)

            grid = rearrange(
                [images, images_reconstructed], "n b c h w -> c (n h) (b w)"
            )
            pl_module.logger.log_image(f"{split}_reconstruction", [grid])

            if is_train:
                pl_module.train()


class VQVAEFCReconstructionLogger(Callback):
    def __init__(self, batch_frequency: int = 10, num_images: int = 10) -> None:
        self.batch_frequency = batch_frequency
        self.num_images = num_images

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        self.log_img(pl_module, batch, batch_idx, split="val")

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if batch_idx % self.batch_frequency == 0:

            is_train = pl_module.training

            if is_train:
                pl_module.eval()

            images = batch[0 : self.num_images]
            images_crops, images_reconstructed, quantize = pl_module(images)
            size = max(pl_module.crop_sizes)

            grid_recon_crops = rearrange(
                [images_crops, images_reconstructed],
                "n (crops b) c h w -> c (crops h) (b n w)",
                b=self.num_images,
            )

            grid_recon_overlap = rearrange(
                [
                    pl_module.cropper.get_overlap(images_crops, size=size),
                    pl_module.cropper.get_overlap(images_reconstructed, size=size),
                ],
                "n b c h w -> c (n h) (b w) ",
            )

            pl_module.logger.log_image(f"{split}_recon_crops", [grid_recon_crops])
            pl_module.logger.log_image(f"{split}_recon_overlap", [grid_recon_overlap])

            if is_train:
                pl_module.train()


class VQVAEFCCodebookDistributionLogger(Callback):
    def __init__(self, num_embeddings: int, batch_frequency: int = 10) -> None:
        self.batch_frequency = batch_frequency
        self.num_embeddings = num_embeddings

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        self.log_histogram(trainer, pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        self.log_histogram(trainer, pl_module, batch, batch_idx, split="val")

    def log_histogram(self, trainer, pl_module, batch, batch_idx, split):
        if batch_idx % self.batch_frequency == 0:
            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            wandb_logger = get_wandb_logger(trainer)
            _, _, quantize = pl_module(batch)
            histogram = np.histogram(
                quantize["encoding_indices"].cpu().numpy(),
                bins=np.arange(self.num_embeddings),
            )

            wandb_logger.experiment.log(
                {
                    f"{split}_codebook_distribution": wandb.Histogram(
                        np_histogram=histogram
                    ),
                }
            )

            if is_train:
                pl_module.train()
