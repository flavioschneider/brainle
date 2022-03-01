from math import sqrt
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, einsum


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


class AttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        block_size: int,
        dropout_attention: float,
        dropout_residual: float,
        use_mask: bool,
    ):
        super().__init__()

        self.block_attention = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            CausalSelfAttention(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                block_size=block_size,
                dropout_attention=dropout_attention,
                dropout_residual=dropout_residual,
                use_mask=use_mask,
            ),
        )

        self.block_mlp = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=4 * embedding_dim),
            nn.GELU(),
            nn.Linear(in_features=4 * embedding_dim, out_features=embedding_dim),
            nn.Dropout(dropout_residual),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.block_attention(x)
        x = x + self.block_mlp(x)
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        block_size: int,
        dropout_embedding: float,
        dropout_attention: float,
        dropout_residual: float,
        use_mask: bool,
    ):
        super().__init__()

        # Embedding
        self.token_embedding = nn.Embedding(
            num_embeddings=vocabulary_size, embedding_dim=embedding_dim
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, block_size, embedding_dim)
        )
        self.dropout = nn.Dropout(dropout_embedding)

        # Attention blocks
        self.blocks = nn.Sequential(
            *[
                AttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    block_size=block_size,
                    dropout_attention=dropout_attention,
                    dropout_residual=dropout_residual,
                    use_mask=use_mask,
                )
                for _ in range(num_layers)
            ]
        )

        # Layer norm and head
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(
            in_features=embedding_dim, out_features=vocabulary_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s = x.shape
        x = self.dropout(self.token_embedding(x) + self.position_embedding)
        x = self.blocks(x)
        x = self.layer_norm(x)
        x = self.head(x)
        return x


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


class SMUNet(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int,
        num_layers: int,
        memory_sizes: List[int],
        num_heads: int,
        kernel_size: int,
        stride: int,
        padding: int,
        use_skip: bool = True,
    ):
        super().__init__()
        comment = "Expected num_layers to equal len(memory_sizes)"
        assert num_layers == len(memory_sizes), comment

        self.num_layers = num_layers
        self.use_skip = use_skip

        self.token_embedding = nn.Embedding(
            num_embeddings=vocabulary_size, embedding_dim=embedding_dim
        )

        encoders = [
            nn.Sequential(
                nn.LayerNorm(embedding_dim),
                SelfMemoryEncode(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    memory_size=memory_sizes[i],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
                nn.LayerNorm(embedding_dim),
                nn.GELU(),
                nn.Linear(in_features=embedding_dim, out_features=embedding_dim),
            )
            for i in range(num_layers)
        ]

        self.encoders = nn.ModuleList(encoders)

        decoders = [
            nn.Sequential(
                nn.LayerNorm(embedding_dim),
                SelfMemoryDecode(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    memory_size=memory_sizes[i],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
                nn.LayerNorm(embedding_dim),
                nn.GELU(),
                nn.Linear(in_features=embedding_dim, out_features=embedding_dim),
            )
            for i in list(reversed(range(num_layers)))
        ]

        self.decoders = nn.ModuleList(decoders)

        self.head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=vocabulary_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s = x.shape
        L = self.num_layers

        x = self.token_embedding(x)
        xs = []

        # Encode
        for i in range(L):
            x = self.encoders[i](x)
            if self.use_skip and i < L - 1:
                xs = [x] + xs

        # Decode
        for i in range(L):
            x = self.decoders[i](x)
            if self.use_skip and i < L - 1:
                x += xs[i]

        return self.head(x)


"""
Attention blocks
"""


class AttentionBase(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, num_heads: int, dropout: float = 0.0
    ):
        super().__init__()
        comment = "Expected in_features and out_features to be divisible by num_heads"
        assert in_features % num_heads == 0 and out_features % num_heads == 0, comment

        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = in_features // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_out = nn.Linear(in_features=out_features, out_features=out_features)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Dimensionaly checks
        d_in, d_out = self.in_features, self.out_features
        (q_b, q_n, q_d), (k_b, k_m, k_d), (v_b, v_m, v_d) = q.shape, k.shape, v.shape
        assert k_b == v_b, "Expected same batch size for k, v"
        assert q_d == k_d == d_in, f"Expected q, k to have {d_in} features"
        assert v_d == d_out, f"Expected v to have {d_out} features"
        assert k_m == v_m, "Expected k, v to have same length"
        # Split heads
        h, hd = self.num_heads, self.head_dim
        q = rearrange(q, "b n (h hd) -> b h n hd", h=h)
        k = rearrange(k, "b m (h hd) -> b h m hd", h=h)
        v = rearrange(v, "b m (h vd) -> b h m vd", h=h)
        # Compute similarty with memory
        sim = einsum("b h i l, b h j l -> b h i j", q, k) * self.scale
        # Compute attention scores
        att = sim.softmax(dim=-1)
        att = self.dropout(att)
        # Compute weighted values
        out = einsum("b h i l, b h l j -> b h i j", att, v)
        out = rearrange(out, "b h n vd -> b n (h vd)", h=h)
        return self.to_out(out)


class SABlock(nn.Module):
    """Self Attention Block"""

    def __init__(
        self, in_features: int, out_features: int, num_heads: int, dropout: float = 0.0
    ):
        super().__init__()
        self.in_features = in_features
        self.attention = AttentionBase(
            in_features=in_features,
            out_features=out_features,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.to_q = nn.Linear(
            in_features=in_features, out_features=in_features, bias=False
        )
        self.to_k = nn.Linear(
            in_features=in_features, out_features=in_features, bias=False
        )
        self.to_v = nn.Linear(
            in_features=in_features, out_features=out_features, bias=False
        )

    def forward(self, x: Tensor) -> Tensor:
        b, n, c = x.shape
        # Dimensionality checks
        assert c == self.in_features, "Expected input of shape [b, n, in_features]"
        # Compute memory attention
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        out = self.attention(q, k, v)
        return out  # [b, n, out_features]


class RABlock(nn.Module):
    """Resize Attention Block"""

    def __init__(
        self,
        in_tokens: int,
        out_tokens: int,
        in_features: int,
        out_features: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_tokens = in_tokens
        self.in_features = in_features
        self.attention = AttentionBase(
            in_features=in_features,
            out_features=out_features,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.q = nn.Parameter(torch.zeros(1, out_tokens, in_features))
        self.q.data.data.normal_()
        self.to_k = nn.Linear(
            in_features=in_features, out_features=in_features, bias=False
        )
        self.to_v = nn.Linear(
            in_features=in_features, out_features=out_features, bias=False
        )

    def forward(self, x: Tensor) -> Tensor:
        b, n, c = x.shape
        # Dimensionality checks
        comment = "Expected input of shape [b, in_tokens, in_features]"
        assert n == self.in_tokens and c == self.in_features, comment
        # Compute resize attention
        q, k, v = self.q, self.to_k(x), self.to_v(x)
        out = self.attention(q, k, v)
        return out  # [b, out_tokens, out_features]


class MABlock(nn.Module):
    """Memory Attention Block"""

    def __init__(
        self,
        memory_size: int,
        in_features: int,
        out_features: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.attention = AttentionBase(
            in_features=in_features,
            out_features=out_features,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.to_q = nn.Linear(
            in_features=in_features, out_features=in_features, bias=False
        )
        self.k = nn.Parameter(torch.zeros(1, memory_size, in_features))
        self.k.data.data.normal_()
        self.v = nn.Parameter(torch.zeros(1, memory_size, out_features))
        self.v.data.data.normal_()

    def forward(self, x: Tensor) -> Tensor:
        b, n, c = x.shape
        # Dimensionality checks
        assert c == self.in_features, "Expected input of shape [b, n, in_features]"
        # Compute memory attention
        q, k, v = self.to_q(x), self.k, self.v
        out = self.attention(q, k, v)
        return out  # [b, n, out_features]
