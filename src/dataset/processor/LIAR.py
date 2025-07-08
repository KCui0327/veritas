"""
Description: This script processes the LIAR dataset by reading multiple TSV files,
concatenating them into a single DataFrame, shuffling the rows, and
removing unnecessary columns.

The final DataFrame is saved to a new CSV file.

Author: Kenny Cui
Date: July 6, 2025
"""

import pandas as pd
import glob
import os
from pathlib import Path

HEADER = [
    "ID",
    "verdict",
    "statement",
    "subject",
    "speaker",
    "speaker_job_title",
    "state_info",
    "party_affiliation",
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
    "context",
]

PROJECT_DIR = Path(__file__).resolve().parents[3]
_DATA_PATH = os.path.join(PROJECT_DIR, 'data')
_OUTPUT_PATH = os.path.join(Path(__file__).resolve().parents[1], 'output')

dataset_files = glob.glob(f"{_DATA_PATH}/LIAR/*.tsv")
dataset = []
for file in dataset_files:
    print(f"Processing {file}")
    df = pd.read_csv(file, sep="\t", header=None, names=HEADER)
    dataset.append(df)

df = pd.concat(dataset, ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)
df.drop(
    columns=[
        "ID",
        "barely_true_counts",
        "false_counts",
        "half_true_counts",
        "mostly_true_counts",
        "pants_on_fire_counts",
        "party_affiliation",
    ],
    inplace=True,
)

remove_val = {"barely-true", "half-true", "mostly-true", "mostly-false"}

# Remove unnecessary columns
del df["speaker_job_title"]
del df["state_info"]
del df["speaker"]

df.rename(columns={"context": "content"}, inplace=True)
df = df[
    ~df["verdict"].isin(remove_val)
]  # Filter out "barely-true", "half-true", and "mostly-true"
df.loc[df["verdict"] == "pants-fire", "verdict"] = (
    "false"  # Merge "pants-fire" into "false"
)
df.loc[df["statement"].notnull(), "statement"] = df[
    "statement"
].str.strip()  # Strip useless characters
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the DataFrame
df["verdict"] = df["verdict"].str.lower()  # Normalize verdicts

df.to_csv(f"{_OUTPUT_PATH}/LIAR.csv", index=False)
print("LIAR dataset processed and saved to LIAR.csv")
