"""
Description: A custom dataset class that contains already processed
word embeddings for phrases in the Veritas dataset.

Author: Kenny Cui
Date: August 3, 2025
"""

import re
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset


class VeritasDatasetEmb(Dataset):
    def __init__(self, dataset_pth: str):
        data = np.load(dataset_pth, allow_pickle=True)
        self.embeddings = data["embeddings"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        input_tensor = torch.tensor(embedding, dtype=torch.float32)
        label = self.labels[idx]
        label_tensor = torch.tensor(1 if label == "True" else 0, dtype=torch.long)

        return input_tensor, label_tensor
