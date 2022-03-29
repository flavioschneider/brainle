from typing import Dict

import faiss
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class KNNBlock(nn.Module):

    """Approxiamte KNN adapted for pyTorch using the Faiss library."""

    def __init__(self, features: int):
        super().__init__()
        self.features = features
        self.index = faiss.IndexFlatL2(features)

    def push(self, x: Tensor) -> int:
        # Convert to numpy
        x = x.cpu().detach().numpy()
        # Dimensionality check
        n, d = x.shape
        assert d == self.features, f"Expected tensor of shape [n, features]"
        # Add to index
        self.index.add(x)
        # Return number of items in the index
        return self.index.ntotal

    def forward(self, x: Tensor, k: int) -> Tensor:
        # Convert to numpy
        x_numpy = x.cpu().detach().numpy()
        # Dimensionality check
        n, d = x.shape
        assert d == self.features, f"Expected tensor of shape [n, features]"
        # KNN search into index with k neighbors
        distances, indices, embedding = self.index.search_and_reconstruct(x_numpy, k)
        return {
            "distances": torch.tensor(distances).to(x),
            "indices": torch.tensor(indices).to(x),
            "embedding": torch.tensor(embedding).to(x),  # Shape [n, k, d]
        }
