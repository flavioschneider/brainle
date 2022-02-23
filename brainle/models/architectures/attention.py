from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        block_size: int,
        dropout_attention: float,
        dropout_residual: float,
        use_mask: bool = True,
    ):
        super().__init__()
        assert (
            embedding_dim % num_heads == 0
        ), "Expected embeddings_dim to be divisible by num_heads"
        self.embedding_dim = embedding_dim
        self.block_size = block_size
        self.head_dim = embedding_dim // num_heads
        self.use_mask = use_mask

        self.query = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.key = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.value = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.head = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)

        self.dropout_attention = nn.Dropout(p=dropout_attention)
        self.dropout_residual = nn.Dropout(p=dropout_residual)

        lower_triangular = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", rearrange(lower_triangular, "s0 s1 -> 1 1 s0 s1"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        assert s == self.block_size, f"Expected {self.block_size} tokens"
        assert d == self.embedding_dim, f"Expected embeddings dim: {self.embedding_dim}"
        hd = self.head_dim
        # Compute query, key, values for all heads
        q, k, v = self.query(x), self.key(x), self.value(x)
        q, k, v = (rearrange(x, "b s (h hd) -> b h s hd", hd=hd) for x in [q, k, v])
        # Compute self attention
        att = torch.einsum("b h i l, b h j l -> b h i j", q, k) / sqrt(hd)
        if self.use_mask:
            att = att.masked_fill(self.mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout_attention(att)
        out = torch.einsum("b h i l, b h l j -> b h i j", att, v)
        # Compute head
        out = self.head(rearrange(out, "b h s hd -> b s (h hd)"))
        out = self.dropout_residual(out)
        return out


"""

Summary of variable abbreviations:

* batch_size = b
* embeddings_dim = d
* num_embeddings = k (vocab size, or number of symbols)
* num_heads = h
* head_dim = hd
* block_size = s (max sequence length)
* matrix multiply indeces: i,j,l

"""
