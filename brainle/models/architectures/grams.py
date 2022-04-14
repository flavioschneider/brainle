from math import sqrt
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, einsum

from .attention import FeedForwardBlock, PositionalEmbedding


class BigramEncoder(nn.Module):
    def __init__(
        self, num_features: int, num_tokens: int, num_nodes: int, num_layers: int = 0
    ):
        super().__init__()
        self.num_features = num_features
        self.num_tokens = num_tokens
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.scale = num_features ** -0.5

        self.to_q = nn.Linear(in_features=num_tokens, out_features=num_nodes)
        self.layers = FeedForwardBlock(num_features, multiplier=2)
        self.to_k = nn.Linear(
            in_features=num_features, out_features=num_features, bias=False
        )
        self.to_v = nn.Linear(
            in_features=num_features, out_features=num_features, bias=False
        )
        self.to_out = nn.Linear(in_features=num_features, out_features=num_features)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        b, t, c = x.shape
        # Dimensionality checks
        assert t == self.num_tokens
        assert c == self.num_features
        x_t = rearrange(x, "b t c -> b c t")
        # Compute q from nodes, k and v from tokens
        q_t, k, v = self.to_q(x_t), self.to_k(x), self.to_v(x)
        q = rearrange(q_t, "b c t -> b t c")
        if self.num_layers > 0:
            q = self.layers(q)
        # Compute similarity to get graph weights
        sim = einsum("b i l, b j l -> b i j", q, k) * self.scale
        # Mask with max values
        amax = torch.amax(sim, dim=1, keepdim=True)
        mask = sim.lt(amax)
        # Get masked attention
        mask_value = -torch.finfo(sim.dtype).max
        sim = sim.masked_fill(mask, mask_value)
        sim = sim - sim.amax(dim=-1, keepdim=True)
        att = sim.softmax(dim=-1)
        att = att.masked_fill(mask, 0)
        # Flow graph forward
        out = einsum("b i l, b l j -> b i j", att, v)
        out = self.to_out(out)
        return out, att


class BigramDecoder(nn.Module):
    def __init__(
        self, num_features: int, num_tokens: int, num_nodes: int, num_layers: int = 0
    ):
        super().__init__()
        self.num_features = num_features
        self.num_tokens = num_tokens
        self.num_nodes = num_nodes
        self.num_layers = num_layers

        self.to_g = nn.Linear(in_features=num_nodes, out_features=num_tokens)
        self.layers = FeedForwardBlock(num_features, multiplier=2)
        self.to_s = nn.Linear(
            in_features=num_features,
            out_features=num_features,
        )

    def forward(self, y: Tensor, att: Tensor) -> Tensor:
        b, n, c = y.shape
        # Dimensionality checks
        assert n == self.num_nodes
        assert c == self.num_features
        # Compute guess nodes and structure nodes
        y_t = rearrange(y, "b n c -> b c n")
        g_t, s = self.to_g(y_t), self.to_s(y)
        g = rearrange(g_t, "b c n -> b n c")
        if self.num_layers > 0:
            g = self.layers(g)
        # Flow graph backwards
        b = einsum("b l i, b l j -> b i j", att, s)
        out = g + b
        return out


class BigramNet(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_tokens: int,
        num_nodes: int,
        vocabulary_size: int,
        use_pos_embedding: bool = False,
        num_layers: int = 0,
    ):
        super().__init__()
        self.use_pos_embedding = use_pos_embedding

        self.token_embedding = nn.Embedding(
            num_embeddings=vocabulary_size, embedding_dim=num_features
        )

        self.positional_embedding = PositionalEmbedding(
            block_size=num_tokens, embedding_dim=num_features
        )

        self.encoder = BigramEncoder(
            num_features=num_features,
            num_tokens=num_tokens,
            num_nodes=num_nodes,
            num_layers=num_layers,
        )

        self.decoder = BigramDecoder(
            num_features=num_features,
            num_tokens=num_tokens,
            num_nodes=num_nodes,
            num_layers=num_layers,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(in_features=num_features, out_features=vocabulary_size),
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        b, s = x.shape
        x = self.token_embedding(x)
        if self.use_pos_embedding:
            x = self.positional_embedding(x)
        y, att = self.encoder(x)
        out = self.decoder(y, att)
        out = self.head(out)
        return {"pred": out, "att": att}
