"""
Description: A custom dataset class for the Veritas dataset.

Author: Kenny Cui
Date: July 6, 2025
"""

import pandas as pd
import torch
import torch.nn as nn
import torchtext
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset


class VeritasDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
    ):
        self.data_frame = pd.read_csv(csv_file)
        self.glove = torchtext.vocab.GloVe(name="6B", dim=300)

        unk_idx = len(self.glove.stoi)
        self.glove.stoi["<UNK>"] = unk_idx
        self.glove.vectors = torch.cat([self.glove.vectors, torch.zeros(1, 300)], dim=0)

        pad_idx = len(self.glove.stoi)
        self.glove.stoi["<PAD>"] = pad_idx
        self.glove.vectors = torch.cat([self.glove.vectors, torch.zeros(1, 300)], dim=0)

        self.embedding = nn.Embedding.from_pretrained(
            self.glove.vectors,
            freeze=True,
            padding_idx=pad_idx,
        )

        self.input_tensors = {}
        self.label_tensors = {}

        self.max_length = 0
        for i, row in self.data_frame.iterrows():
            statement_length = len(row["statement"].split())
            self.max_length = max(self.max_length, statement_length)

        for i, row in self.data_frame.iterrows():
            statement_embedding = self.preprocess_statement(row["statement"])
            self.input_tensors[i] = statement_embedding
            self.label_tensors[i] = torch.tensor(
                0 if row["verdict"] else 1, dtype=torch.float
            )

    def preprocess_statement(self, statement: str) -> torch.Tensor:
        indices = []
        for word in statement.split():
            if word not in self.glove.stoi:
                indices.append(self.glove.stoi["<UNK>"])
            else:
                indices.append(self.glove.stoi[word])

        if len(indices) < self.max_length:
            indices.extend(
                [self.glove.stoi["<PAD>"]] * (self.max_length - len(indices))
            )

        indices_tensor = torch.tensor(indices, dtype=torch.long)
        embedded_sentence = self.embedding(indices_tensor)
        return embedded_sentence

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        return self.input_tensors[idx], self.label_tensors[idx]
