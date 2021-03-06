import io

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
import torchvision
import wandb
from einops import rearrange
from PIL import Image
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


class TextLogger(Callback):
    def __init__(self, batch_frequency: int = 10) -> None:
        self.batch_frequency = batch_frequency
        self.count = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_txt(trainer, pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        self.log_txt(trainer, pl_module, batch, batch_idx, split="val")

    def log_txt(self, trainer, pl_module, batch, batch_idx, split):
        if batch_idx % self.batch_frequency == 0:
            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            dataset = trainer.datamodule.dataset
            wandb_logger = get_wandb_logger(trainer)
            sequence, mask = batch
            out = F.softmax(pl_module(sequence, mask), dim=-1)

            batch_size = out.shape[0]
            ids = torch.topk(out, k=1, dim=-1)[1]
            ids = rearrange(ids, "b s 1 -> b s")

            text_table = wandb.Table(columns=["id", "text", "text_masked", "text_pred"])

            for i in range(batch_size):
                text = "".join(
                    dataset.decode(sequence[i].detach().cpu().numpy().tolist())
                )
                text_masked = "".join(
                    [char if mask[i][idx] else "???" for idx, char in enumerate(text)]
                )
                text_pred = "".join(
                    dataset.decode(ids[i].detach().cpu().numpy().tolist())
                )
                text_table.add_data(f"{self.count}_{i}", text, text_masked, text_pred)

            wandb_logger.experiment.log({"text_table": text_table})
            self.count += 1

            if is_train:
                pl_module.train()


class BigramLogger(Callback):
    def __init__(self, batch_frequency: int = 50) -> None:
        self.batch_frequency = batch_frequency
        self.count = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_txt(trainer, pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        self.log_txt(trainer, pl_module, batch, batch_idx, split="val")

    def plot_attention_matrix(self, matrix, labels):
        w, h = matrix.shape
        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                text=[labels] * h,
                texttemplate="%{text}",
                textfont={"size": 20},
            )
        )
        fig_bytes = fig.to_image(format="png")
        buf = io.BytesIO(fig_bytes)
        return Image.open(buf)

    def log_txt(self, trainer, pl_module, batch, batch_idx, split):
        if batch_idx % self.batch_frequency == 0:
            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            dataset = trainer.datamodule.dataset
            wandb_logger = get_wandb_logger(trainer)
            sequence, mask = batch
            out = pl_module(sequence, mask)
            distribution = F.softmax(out["pred"], dim=-1)

            batch_size = distribution.shape[0]
            ids = torch.topk(distribution, k=1, dim=-1)[1]
            ids = rearrange(ids, "b s 1 -> b s")

            text_table = wandb.Table(
                columns=["id", "text", "text_pred", "matrix", "mask"]
            )

            for i in range(10):
                text = "".join(
                    dataset.decode(sequence[i].detach().cpu().numpy().tolist())
                )
                text_pred = "".join(
                    dataset.decode(ids[i].detach().cpu().numpy().tolist())
                )
                # Get attention matrix image with text overlay
                matrix = out["att"][i].detach().cpu().numpy()
                matrix_image = self.plot_attention_matrix(matrix, list(text))
                # Get mask image with text_pred overlay
                mask = out["att"][i].gt(0).float().detach().cpu().numpy()
                mask_image = self.plot_attention_matrix(mask, list(text_pred))

                text_table.add_data(
                    f"{self.count}_{i}",
                    text,
                    text_pred,
                    wandb.Image(matrix_image),
                    wandb.Image(mask_image),
                )

            wandb_logger.experiment.log({"text_table": text_table})
            self.count += 1

            if is_train:
                pl_module.train()
