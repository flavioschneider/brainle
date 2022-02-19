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
        self.num_embeddings = num_embeddings  # [K]
        self.embedding_dim = embedding_dim  # [D]
        self.beta = beta

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: torch.Tensor):
        B, D, H, W = z.shape
        assert D == self.embedding_dim

        z_flat = rearrange(z, "B D H W -> (B H W) D")  # [N, D]
        codebook = self.embedding.weight  # [K, D]

        distances = (
            reduce(z_flat ** 2, "N D -> N 1", "sum")
            + reduce(codebook ** 2, "K D -> K", "sum")
            - 2
            * torch.einsum("N D, D K -> N K", z_flat, rearrange(codebook, "K D -> D K"))
        )  # [N, K]

        encoding_indices = torch.argmin(distances, dim=1)  # [N]
        encodings = F.one_hot(encoding_indices, num_classes=self.num_embeddings).to(
            z
        )  # [N, K]
        encodings_mean = reduce(encodings, "N K -> K", "mean")
        perplexity = torch.exp(
            -torch.sum(encodings_mean * torch.log(encodings_mean + 1e-10))
        )

        z_quantized_flat = self.embedding(encoding_indices)  # [N, D]
        z_quantized = rearrange(z_quantized_flat, "(B H W) D -> B D H W", B=B, W=W, H=H)

        loss_encoder = F.mse_loss(
            z_quantized.detach(), z
        )  # Force encoder output to match codebook
        loss_codebook = F.mse_loss(
            z_quantized, z.detach()
        )  # Force codebook to match encoder output
        loss = self.beta * loss_encoder + loss_codebook

        z_quantized = (
            z + (z_quantized - z).detach()
        )  # To preserve gradients through codebook

        return {
            "quantized": z_quantized,
            "loss": loss,
            "perplexity": perplexity,
            "encodings": encodings,
            "encoding_indices": encoding_indices,
            "distances": distances,
        }
