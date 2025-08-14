"""
Description: A custom dataset class for the Veritas dataset.

Author: Kenny Cui
Date: July 6, 2025
"""

import re

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchtext.vocab import GloVe


class VeritasDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
    ):
        self.data_frame = pd.read_csv(csv_file)
        self.glove = GloVe(name="6B", dim=50)

        self.input_tensors = {}
        self.label_tensors = {}

        for i, row in self.data_frame.iterrows():
            statement_embedding = self.preprocess_statement(row["statement"])
            self.input_tensors[i] = statement_embedding
            self.label_tensors[i] = torch.tensor(
                0 if row["verdict"] else 1, dtype=torch.float
            )

    def preprocess_statement(self, statement: str) -> torch.Tensor:
        tokens = []

        for token in statement.split():
            token = token.lower()
            clean = re.sub(r"[^A-Za-z\s]", "", token)
            tokens.append(clean)

        tokens = self.glove.get_vecs_by_tokens(tokens, lower_case_backup=True)
        return tokens

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        return self.input_tensors[idx], self.label_tensors[idx]
