from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CausalSelfAttention(nn.Module):
    """

    Summary of variable abbreviations:

    * batch_size = b
    * embeddings_dim = d
    * num_embeddings (vocab size, or number of symbols)
    * num_heads = h
    * head_dim = hd
    * block_size = s (max sequence length)
    * matrix multiply indeces: i,j,l

    """

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
        comment = "Expected embeddings_dim to be divisible by num_heads"
        assert embedding_dim % num_heads == 0, comment

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


class SelfMemoryEncode(nn.Module):
    """
    Summary of variable abbreviations:

    * embeddings_dim = d
    * batch_size = b
    * memory_size = m
    * kernel_size = k
    * stride = st
    * padding = p
    * num_heads = h
    * head_dim = hd
    * value_dim = vd
    * block_size = s
    * window_blocks = w
    * num_embeddings (vocab size, or number of symbols)
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        memory_size: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()

        comment = "Expected embeddings_dim to be divisible by num_heads"
        assert embedding_dim % num_heads == 0, comment
        comment = "Expected embedding_dim to be divisible by (kernel_size * num_heads)"
        assert embedding_dim % (kernel_size * num_heads) == 0, comment

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.value_dim = embedding_dim // (kernel_size * num_heads)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.query = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.keys = nn.Linear(in_features=self.head_dim, out_features=memory_size)
        self.values = nn.Linear(in_features=memory_size, out_features=self.value_dim)
        self.head = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        assert d == self.embedding_dim, f"Expected embeddings dim: {self.embedding_dim}"
        hd, h = self.head_dim, self.num_heads
        k, st, p = self.kernel_size, self.stride, self.padding
        # Compute queries
        q = self.query(x)
        # Slide window to get w blocks of k tokens embeddings and merge with batch dim
        q = rearrange(q, "b s d -> b d s 1")
        q = F.unfold(q, kernel_size=(k, 1), stride=(st, 1), padding=(p, 0))
        q = rearrange(q, "b (d k) w -> (b w k) d", d=d, k=k)
        # Split heads
        q = rearrange(q, "bwk (h hd) -> (bwk h) hd", hd=hd)
        # Attend queries to memory
        att = F.softmax(self.keys(q) / sqrt(hd), dim=-1)
        att = self.values(att)
        # Merge all heads in window and compute
        heads = rearrange(att, "(bw k h) vd -> bw (k h vd)", k=k, h=h)
        out = self.head(heads)
        out = rearrange(out, "(b w) d -> b w d", b=b)
        return out


class SelfMemoryDecode(nn.Module):
    """
    Summary of variable abbreviations:

    * embeddings_dim = d
    * batch_size = b
    * memory_size = m
    * kernel_size = k
    * stride = st
    * padding = p
    * num_heads = h
    * head_dim = hd
    * value_dim = vd
    * block_size = s
    * output_size = o
    * num_embeddings (vocab size, or number of symbols)
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        memory_size: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()

        comment = "Expected embeddings_dim to be divisible by num_heads"
        assert embedding_dim % num_heads == 0, comment
        comment = "Expected embedding_dim to be divisible by (kernel_size * num_heads)"
        assert embedding_dim % (kernel_size * num_heads) == 0, comment

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // (kernel_size * num_heads)
        self.value_dim = embedding_dim // num_heads
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.query = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.keys = nn.Linear(in_features=self.head_dim, out_features=memory_size)
        self.values = nn.Linear(in_features=memory_size, out_features=self.value_dim)
        self.head = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        assert d == self.embedding_dim, f"Expected embeddings dim: {self.embedding_dim}"
        k, st, p = self.kernel_size, self.stride, self.padding
        hd, h = self.head_dim, self.num_heads
        # Compute queries
        q = self.query(x)
        # Split heads
        q = rearrange(q, "b s (k h hd) -> (b s k h) hd", k=k, hd=hd)
        # Attend queries to memory
        att = F.softmax(self.keys(q) / sqrt(hd), dim=-1)
        att = self.values(att)
        # Fold to tokens and average overlaps
        att = rearrange(att, "(b s k h) vd -> b (k h vd) s", b=b, s=s, k=k, h=h)
        output_size = ((s - 1) * st + k - 2 * p, 1)
        fold = nn.Fold(output_size, kernel_size=(k, 1), stride=(st, 1), padding=(p, 0))
        out = fold(att) / fold(torch.ones_like(att))
        out = rearrange(out, "b d o 1 -> b o d")
        # Compute head
        out = self.head(out)
        return out
