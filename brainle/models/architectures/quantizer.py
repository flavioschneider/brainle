from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import Tensor, einsum


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        beta: float = 0.25,  # Encoder loss weight
    ):
        super().__init__()
        self.num_embeddings = num_embeddings  # [k]
        self.embedding_dim = embedding_dim  # [d]
        self.beta = beta
        # Embedding parameters
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: torch.Tensor):
        b, d, h, w = z.shape
        k = self.num_embeddings
        assert d == self.embedding_dim, "Input channels must equal embedding_dim."

        z_flat = rearrange(z, "b d h w -> (b h w) d")  # [n, d]
        embedding = self.embedding.weight  # [k, d]

        distances = (
            reduce(z_flat ** 2, "n d -> n 1", "sum")
            + reduce(embedding ** 2, "k d -> k", "sum")
            - 2 * torch.einsum("n d, d k -> n k", z_flat, embedding.t())
        )  # [n, k]

        encoding_indices = torch.argmin(distances, dim=1)  # [n]
        encodings_onehot = F.one_hot(encoding_indices, num_classes=k).float()  # [n, k]

        z_quantized_flat = self.embedding(encoding_indices)  # [n, d]
        z_quantized = rearrange(z_quantized_flat, "(b h w) d -> b d h w", b=b, w=w, h=h)

        # Force encoder output to match codebook
        loss_encoder = F.mse_loss(z_quantized.detach(), z)
        # Force codebook to match encoder output
        loss_codebook = F.mse_loss(z_quantized, z.detach())
        loss = self.beta * loss_encoder + loss_codebook
        # To preserve gradients through embedding codebook
        z_quantized = z + (z_quantized - z).detach()

        return {
            "quantized": z_quantized,
            "loss": loss,
            "perplexity": self.eval_perplexity(encodings_onehot),
            "encodings_onehot": encodings_onehot,
            "encoding_indices": encoding_indices,
            "distances": distances,
        }

    def eval_perplexity(self, encodings_onehot: torch.Tensor) -> torch.Tensor:
        encodings_mean = reduce(encodings_onehot.float(), "n k -> k", "mean")
        perplexity = torch.exp(
            -torch.sum(encodings_mean * torch.log(encodings_mean + 1e-10))
        )
        return perplexity


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        beta: float = 0.25,  # Encoder loss weight
        ema_decay: float = 0.99,
        ema_epsilon: float = 1e-5,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings  # [k]
        self.embedding_dim = embedding_dim  # [d]
        self.beta = beta
        self.ema_decay = ema_decay
        self.ema_epsilon = ema_epsilon
        # Embedding parameters
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.normal_()
        # Exponential Moving Average (EMA) parameters
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_embedding_avg", self.embedding.weight.clone())

    def forward(self, z: torch.Tensor):
        b, d, h, w = z.shape
        k = self.num_embeddings
        assert d == self.embedding_dim

        z_flat = rearrange(z, "b d h w -> (b h w) d")  # [n, d]
        embedding = self.embedding.weight  # [k, d]

        distances = (
            reduce(z_flat ** 2, "n d -> n 1", "sum")
            + reduce(embedding ** 2, "k d -> k", "sum")
            - 2 * torch.einsum("n d, d k -> n k", z_flat, embedding.t())
        )  # [n, k]

        encoding_indices = torch.argmin(distances, dim=1)  # [n]
        encodings_onehot = F.one_hot(encoding_indices, num_classes=k).float()  # [n, k]

        z_quantized_flat = self.embedding(encoding_indices)  # [n, d]
        z_quantized = rearrange(z_quantized_flat, "(b h w) d -> b d h w", b=b, w=w, h=h)

        # Force encoder output to match embedding
        loss_encoder = F.mse_loss(z_quantized.detach(), z)
        loss = self.beta * loss_encoder

        # Update embedding with EMA
        if self.training:
            self.update_embedding(z_flat, encodings_onehot)

        # To preserve gradients through embedding codebook
        z_quantized = z + (z_quantized - z).detach()

        return {
            "quantized": z_quantized,
            "loss": loss,
            "perplexity": self.eval_perplexity(encodings_onehot),
            "encodings_onehot": encodings_onehot,
            "encoding_indices": encoding_indices,
            "distances": distances,
        }

    def eval_perplexity(self, encodings_onehot: torch.Tensor) -> torch.Tensor:
        encodings_mean = reduce(encodings_onehot.float(), "n k -> k", "mean")
        perplexity = torch.exp(
            -torch.sum(encodings_mean * torch.log(encodings_mean + 1e-10))
        )
        return perplexity

    def update_embedding(
        self, z_flat: torch.Tensor, encodings_onehot: torch.Tensor
    ) -> None:
        k = self.num_embeddings

        batch_cluster_size = reduce(encodings_onehot, "n k -> k", "sum")
        batch_embedding_avg = torch.einsum(
            "k n, n d -> k d", encodings_onehot.t(), z_flat
        )
        self.ema_cluster_size.data.mul_(self.ema_decay).add_(
            batch_cluster_size, alpha=1 - self.ema_decay
        )  # [k]
        self.ema_embedding_avg.data.mul_(self.ema_decay).add_(
            batch_embedding_avg, alpha=1 - self.ema_decay
        )
        new_embedding = self.ema_embedding_avg / rearrange(
            self.ema_cluster_size + 1e-5, "k -> k 1"
        )
        self.embedding.weight.data.copy_(new_embedding)


"""
    New unopinionated version
"""


class QuantizerBase(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q: Tensor, k: Tensor) -> Tensor:
        (n, c), (m, c_) = q.shape, k.shape
        # Dimensionality checks
        assert c == c_, "Expected q, k to have same number of channels"
        # Compute similarity between queries and value vectors
        similarity = self.similarity(q, k)  # [n, m]
        # Get quatized indeces
        z_indices = torch.argmin(similarity, dim=1)  # [n]
        z_onehot = F.one_hot(z_indices, num_classes=m).float()  # [n, m]
        # Get quantized vectors
        z = einsum("n m, m c -> n c", z_onehot, k)
        # Copy gradients to input
        z = q + (z - q).detach()
        return {
            "embedding": z,
            "indices": z_indices,
            "onehot": z_onehot,
            "perplexity": self.perplexity(z_onehot),
        }

    def similarity(self, q: Tensor, k: Tensor) -> Tensor:
        l2_q = reduce(q ** 2, "n c -> n 1", "sum")
        l2_k = reduce(k ** 2, "m c -> m", "sum")
        sim = einsum("n c, m c -> n m", q, k)
        return l2_q + l2_k - 2 * sim

    def perplexity(self, z_onehot: Tensor) -> Tensor:
        z_mean = reduce(z_onehot, "n m -> m", "mean")
        perplexity = torch.exp(-torch.sum(z_mean * torch.log(z_mean + 1e-10)))
        return perplexity


class MQBlock(nn.Module):
    """Memory Quantization Block with EMA"""

    def __init__(
        self,
        features: int,
        memory_size: int,
        ema_decay: float = 0.99,
        ema_epsilon: float = 1e-5,
    ):
        super().__init__()
        self.memory_size = memory_size
        self.quantizer = QuantizerBase()
        self.ema_decay = ema_decay
        self.ema_epsilon = ema_epsilon
        # Embedding parameters
        self.embedding = nn.Embedding(memory_size, features)
        self.embedding.weight.data.normal_()
        # Exponential Moving Average (EMA) parameters
        self.register_buffer("ema_cluster_size", torch.zeros(memory_size))
        self.register_buffer("ema_embedding_avg", self.embedding.weight.clone())

    def forward(self, x: Tensor) -> Tensor:
        b, n, c = x.shape
        # Flatten
        q = rearrange(x, "b n c -> (b n) c")
        # Compute quantization
        k = self.embedding.weight
        z = self.quantizer(q, k)
        # Update embedding with EMA
        if self.training:
            self.update_embedding(q, z["onehot"])
        # Unflatten all and return
        return {
            "embedding": rearrange(z["embedding"], "(b n) c -> b n c", b=b),
            "indices": rearrange(z["indices"], "(b n) -> b n", b=b),
            "onehot": rearrange(z["onehot"], "(b n) m -> b n m", b=b),
            "perplexity": z["perplexity"],
        }

    def update_embedding(self, q: Tensor, z_onehot: Tensor) -> None:
        # Moves selected embeddings towards q using EMA

        batch_cluster_size = reduce(z_onehot, "n m -> m", "sum")
        batch_embedding_avg = einsum("n m, n c -> m c", z_onehot, q)
        self.ema_cluster_size.data.mul_(self.ema_decay).add_(
            batch_cluster_size, alpha=1 - self.ema_decay
        )  # [m]
        self.ema_embedding_avg.data.mul_(self.ema_decay).add_(
            batch_embedding_avg, alpha=1 - self.ema_decay
        )
        new_embedding = self.ema_embedding_avg / rearrange(
            self.ema_cluster_size + 1e-5, "k -> k 1"
        )
        self.embedding.weight.data.copy_(new_embedding)


class MultiMQ(nn.Module):

    """Memory Quantization with many codebooks."""

    def __init__(
        self,
        channels_list: List[int],
        memory_size: int,
        ema_decay: float = 0.99,
        ema_epsilon: float = 1e-5,
    ):
        super().__init__()
        self.channels_list = channels_list
        self.tot_channels = sum(channels_list)
        self.num_blocks = len(self.channels_list)
        self.channels_list += [0]  # To avoid loop overflow

        self.blocks = nn.ModuleList(
            [
                MQBlock(
                    features=channels,
                    memory_size=memory_size,
                    ema_decay=ema_decay,
                    ema_epsilon=ema_epsilon,
                )
                for channels in channels_list
            ]
        )

    def forward(self, z: Tensor):
        b, n, c = z.shape
        assert c == self.tot_channels, f"Expected channels_list to sum to {c}."

        quantized = []
        head, tail = 0, self.channels_list[0]

        # Split channels into chunks and feead each into a different MQ block.
        for i in range(self.num_blocks):
            quantized += [self.blocks[i](z[:, :, head:tail])]
            head = tail
            tail = tail + self.channels_list[i + 1]

        return {
            "embedding": torch.cat([q["embedding"] for q in quantized], dim=2),
            "indices": torch.cat([q["indices"] for q in quantized], dim=1),
            "onehot": torch.cat([q["onehot"] for q in quantized], dim=1),
            "perplexity": torch.tensor([q["perplexity"] for q in quantized]),
        }
