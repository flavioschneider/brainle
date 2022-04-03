from typing import List

import numpy as np
import torch
from tokenizers import normalizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import Dataset


class MaskedCharDataset(Dataset):
    def __init__(
        self, text: str, block_size: int, p_word_mask: float, p_char_mask: float
    ):
        normalizer = normalizers.Sequence([NFD(), StripAccents()])
        text = normalizer.normalize_str(text)
        text_length = len(text)
        alphabet = sorted(list(set(text)))
        print(f"Alphabet size {len(alphabet) + 1}")

        # Keep index 0 for mask
        self.char_to_id = {char: idx + 1 for idx, char in enumerate(alphabet)}
        self.id_to_char = {idx + 1: char for idx, char in enumerate(alphabet)}
        self.id_to_char[0] = "▢"
        self.id_to_char["▢"] = 0

        self.block_size = block_size
        self.alphabet = alphabet
        self.text = text
        self.p_word_mask = p_word_mask
        self.p_char_mask = p_char_mask
        self.pre_tokenizer = Whitespace()

    def __len__(self):
        return len(self.text) // self.block_size

    def encode(self, text: str) -> List[int]:
        return [self.char_to_id[char] for char in text]

    def decode(self, ids: List[int]) -> str:
        return "".join([self.id_to_char[idx] for idx in ids])

    def __getitem__(self, idx):
        chunk = self.text[
            idx * self.block_size : idx * self.block_size + self.block_size + 1
        ]

        x = torch.tensor(self.encode(chunk), dtype=torch.long)[0 : self.block_size]
        mask = torch.ones_like(x)

        # Mask random words
        words = self.pre_tokenizer.pre_tokenize_str(chunk)
        words_mask = np.random.binomial(1, self.p_word_mask, len(words))
        for idx, word_triplet in enumerate(words):
            word, (start, end) = word_triplet
            if words_mask[idx]:
                mask[start:end] = 0

        # Mask random letters
        chars_mask = torch.tensor(
            np.random.binomial(1, self.p_char_mask, self.block_size)
        )
        mask[chars_mask == 1] = 0

        return x, mask
