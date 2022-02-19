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

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: torch.Tensor):
        b, d, h, w = z.shape
        assert d == self.embedding_dim

        z_flat = rearrange(z, "b d h w -> (b h w) d")  # [n, d]
        codebook = self.embedding.weight  # [k, d]

        distances = (
            reduce(z_flat ** 2, "n d -> n 1", "sum")
            + reduce(codebook ** 2, "k d -> k", "sum")
            - 2
            * torch.einsum("n d, d k -> n k", z_flat, rearrange(codebook, "k d -> d k"))
        )  # [n, k]

        encoding_indices = torch.argmin(distances, dim=1)  # [n]
        encodings = F.one_hot(
            encoding_indices, num_classes=self.num_embeddings
        )  # [n, k]
        encodings_mean = reduce(encodings, "n k -> k", "mean")
        perplexity = torch.exp(
            -torch.sum(encodings_mean * torch.log(encodings_mean + 1e-10))
        )

        z_quantized_flat = self.embedding(encoding_indices)  # [n, d]
        z_quantized = rearrange(z_quantized_flat, "(b h w) d -> b d h w", b=b, w=w, h=h)

        # Force encoder output to match codebook
        loss_encoder = F.mse_loss(z_quantized.detach(), z)
        # Force codebook to match encoder output
        loss_codebook = F.mse_loss(z_quantized, z.detach())
        loss = self.beta * loss_encoder + loss_codebook
        # To preserve gradients through codebook
        z_quantized = z + (z_quantized - z).detach()

        return {
            "quantized": z_quantized,
            "loss": loss,
            "perplexity": perplexity,
            "encodings": encodings,
            "encoding_indices": encoding_indices,
            "distances": distances,
        }
