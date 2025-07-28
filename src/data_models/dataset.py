"""
Description: A custom dataset class for the Veritas dataset.

Author: Kenny Cui
Date: July 6, 2025
"""

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset


class VeritasDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
    ):
        self.data_frame = pd.read_csv(csv_file)
        self.embedding = SentenceTransformer("all-MiniLM-L6-v2")
        self.inputs = {}
        self.labels = {}
        self._build_vocaibulary()

    def _build_vocabulary(self):
        for i, row in self.data_frame.iterrows():
            sequence_tensors = self.embedding.encode(
                sentences=row["statement"])
            self.inputs[i] = sequence_tensors
            self.labels[i] = torch.tensor(0 if row["verdict"] else 1, dtype=torch.float)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]
