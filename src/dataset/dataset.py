"""
Description: A custom dataset class for the Veritas dataset.

Author: Kenny Cui
Date: July 6, 2025
"""

import re
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset


class VeritasDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        vocab_size: int = 20000,
        max_length: int = 512,
    ):
        self.data_frame = pd.read_csv(csv_file)

        self.vocab_size = vocab_size
        self.max_length = max_length
        self.word2idx = None
        self.idx2word = None

        # Build vocabulary and convert labels
        self._build_vocabulary()

    def _preprocess_text(self, text):
        """Clean and tokenize text."""
        if not isinstance(text, str):
            return []

        text = re.sub(r"[^\w\s]", "", text.lower())
        tokens = text.split()

        return tokens

    def _build_vocabulary(self):
        """Build vocabulary from all texts in the dataset."""
        word_counts = Counter()

        for _, row in self.data_frame.iterrows():
            tokens = self._preprocess_text(row["statement"])
            word_counts.update(tokens)

        # Get most common words
        most_common = word_counts.most_common(
            self.vocab_size - 2
        )  # -2 for <PAD> and <UNK>

        # Create word2idx mapping
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.word2idx.update(
            {word: idx + 2 for idx, (word, _) in enumerate(most_common)}
        )

        # Create idx2word mapping
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def _text_to_tensor(self, text):
        """Convert text to tensor of token indices."""
        tokens = self._preprocess_text(text)

        # Convert tokens to indices
        indices = []
        for token in tokens:
            if token in self.word2idx:
                indices.append(self.word2idx[token])
            else:
                indices.append(self.word2idx["<UNK>"])

        # Pad or truncate to max_length
        if len(indices) < self.max_length:
            indices.extend([self.word2idx["<PAD>"]] * (self.max_length - len(indices)))
        else:
            indices = indices[: self.max_length]

        return torch.tensor(indices, dtype=torch.long)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        input_tensor = self._text_to_tensor(self.data_frame["statement"][idx])
        label_tensor = torch.tensor(
            0 if self.data_frame["verdict"][idx] == "True" else 1, dtype=torch.float
        )

        return input_tensor, label_tensor
