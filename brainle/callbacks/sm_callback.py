import numpy as np
import torch
import torch.nn.functional as F
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


class TextLogger(Callback):
    def __init__(self, batch_frequency: int = 10) -> None:
        self.batch_frequency = batch_frequency
        self.text_table = wandb.Table(columns=["id", "text_masked", "text"])

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
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

            for i in range(batch_size):
                text = "".join(
                    dataset.decode(sequence[i].detach().cpu().numpy().tolist())
                )
                text = "".join(
                    [char if mask[i][idx] else "▢" for idx, char in enumerate(text)]
                )
                text_pred = "".join(
                    dataset.decode(ids[i].detach().cpu().numpy().tolist())
                )
                self.text_table.add_data(f"{i}", text, text_pred)

            wandb_logger.experiment.log({"text_table": self.text_table})

            if is_train:
                pl_module.train()
