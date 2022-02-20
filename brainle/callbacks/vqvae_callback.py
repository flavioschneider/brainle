import torch
import torchvision
from einops import rearrange
from pytorch_lightning.callbacks import Callback


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

            grid = rearrange(
                [images_crops, images_reconstructed],
                "n b (c crops) h w -> c (crops h) (b n w)",
                c=3,
            )
            pl_module.logger.log_image(f"{split}_reconstruction", [grid])

            if is_train:
                pl_module.train()
