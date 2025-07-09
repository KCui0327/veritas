"""
Description: A custom dataset class for the Veritas dataset.

Author: Kenny Cui
Date: July 6, 2025
"""

from torch.utils.data import Dataset
import pandas as pd


class VeritasDataset(Dataset):
    def __init__(self, csv_file, statements=None, verdicts=None):
        # Allow for init. with either csv or statements and verdicts arrays
        if not statements and not verdicts:
            self.data_frame = pd.read_csv(csv_file)
        elif statements and verdicts:
            if len(statements) != len(verdicts):
                raise ValueError("Statements and verdicts must have the same length")
            self.data_frame = pd.DataFrame(
                {"statement": statements, "verdict": verdicts}
            )
        else:
            raise ValueError("Both statements and verdicts must be provided or neither")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Can handle both integer indices and slices

        if isinstance(idx, slice):
            samples = self.data_frame.iloc[idx]
            statements, verdicts = [], []

            for _, sample in samples.iterrows():
                statements.append(sample["statement"])
                verdicts.append(sample["verdict"])
            return statements, verdicts

        if not isinstance(idx, int):
            raise TypeError("Index must be an integer or a slice")

        if idx >= len(self.data_frame):
            raise IndexError("Index out of bounds for dataset")

        sample = self.data_frame.iloc[idx]

        return sample["statement"], sample["verdict"]
