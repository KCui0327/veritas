"""
Description: A custom dataset class for the Veritas dataset.

Author: Kenny Cui
Date: July 6, 2025
"""

import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class VeritasDataset(Dataset):
    def __init__(
        self, csv_file, statements=None, verdicts=None, vocab_size=20000, max_length=512
    ):
        # Allow for init. with either csv or statements and verdicts arrays
        if not statements and not verdicts:
            # Use absolute path if csv_file is relative
            if not os.path.isabs(csv_file):
                current_dir = os.path.dirname(os.path.abspath(__file__))
                csv_file = os.path.join(current_dir, csv_file)
            self.data_frame = pd.read_csv(csv_file)
        elif statements and verdicts:
            if len(statements) != len(verdicts):
                raise ValueError("Statements and verdicts must have the same length")
            self.data_frame = pd.DataFrame(
                {"statement": statements, "verdict": verdicts}
            )
        else:
            raise ValueError("Both statements and verdicts must be provided or neither")

        self.vocab_size = vocab_size
        self.max_length = max_length
        self.word2idx = None
        self.idx2word = None

        # Build vocabulary and convert labels
        self._build_vocabulary()
        self._convert_labels()

    def _preprocess_text(self, text):
        """Clean and tokenize text."""
        if not isinstance(text, str):
            return []

        # Convert to lowercase and remove special characters
        text = re.sub(r"[^\w\s]", "", text.lower())
        # Split into words
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

    def _convert_labels(self):
        """Convert text labels to numeric values."""
        # Get unique labels
        unique_labels = self.data_frame["verdict"].unique()

        # Create label mapping (assuming binary classification: fake=1, real=0)
        label_mapping = {}
        for label in unique_labels:
            label_lower = str(label).lower()
            if any(
                fake_word in label_lower for fake_word in ["fake", "false", "0", "no"]
            ):
                label_mapping[label] = 1  # Fake news
            else:
                label_mapping[label] = 0  # Real news

        # Convert labels
        self.data_frame["label"] = self.data_frame["verdict"].map(label_mapping)

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
        # Can handle both integer indices and slices
        if isinstance(idx, slice):
            samples = self.data_frame.iloc[idx]
            inputs, labels = [], []

            for _, sample in samples.iterrows():
                inputs.append(self._text_to_tensor(sample["statement"]))
                labels.append(torch.tensor(sample["label"], dtype=torch.float))

            return torch.stack(inputs), torch.stack(labels)

        if not isinstance(idx, int):
            raise TypeError("Index must be an integer or a slice")

        if idx >= len(self.data_frame):
            raise IndexError("Index out of bounds for dataset")

        sample = self.data_frame.iloc[idx]

        # Convert text to tensor and label to tensor
        input_tensor = self._text_to_tensor(sample["statement"])
        label_tensor = torch.tensor(sample["label"], dtype=torch.float)

        return input_tensor, label_tensor
