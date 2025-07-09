"""
Description: A custom dataset class for the Veritas dataset.

Author: Kenny Cui
Date: July 6, 2025
"""

from torch.utils.data import Dataset
import pandas as pd


class VeritasDataset(Dataset):
    def __init__(self, csv_file):
        self.data_frame = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            samples = self.data_frame.iloc[idx]
            ret = []

            for _, sample in samples.iterrows():
                ret.append(
                    {"statement": sample["statement"], "verdict": sample["verdict"]}
                )
            return ret

        if not isinstance(idx, int):
            raise TypeError("Index must be an integer or a slice")

        if idx >= len(self.data_frame):
            raise IndexError("Index out of bounds for dataset")

        sample = self.data_frame.iloc[idx]

        sample = {"statement": sample["statement"], "verdict": sample["verdict"]}

        return sample
