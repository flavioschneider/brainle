import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce


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

        # Update embedding with EMA
        if self.training:
            self.update_embedding(z_flat, encodings_onehot)

        # Force encoder output to match embedding
        loss_encoder = F.mse_loss(z_quantized.detach(), z)
        loss = self.beta * loss_encoder
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

        self.ema_cluster_size = self.ema_decay * self.ema_cluster_size + (
            1 - self.ema_decay
        ) * reduce(
            encodings_onehot, "n k -> k", "sum"
        )  # [k]

        encodings_sum = torch.einsum("k n, n d -> k d", encodings_onehot.t(), z_flat)
        self.ema_embedding_avg = (
            self.ema_decay * self.ema_embedding_avg
            + (1 - self.ema_decay) * encodings_sum
        )

        ema_cluster_sum = self.ema_cluster_size.sum()
        self.ema_cluster_size = (
            (self.ema_epsilon + self.ema_cluster_size)
            / (ema_cluster_sum + self.ema_epsilon * k)
        ) * ema_cluster_sum
        embedding_normalized = self.ema_embedding_avg / rearrange(
            self.ema_cluster_size, "k -> k 1"
        )

        self.embedding.weight.data.copy_(embedding_normalized)
