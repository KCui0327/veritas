"""
Description: This script processes the WELFake dataset by reading a CSV file,

The final DataFrame is saved to a new CSV file.

Author: Kenny Cui
Date: July 6, 2025
"""

import os
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[3]
_DATA_PATH = os.path.join(PROJECT_DIR, "data")
_OUTPUT_PATH = os.path.join(Path(__file__).resolve().parents[1], "output")

columns = ["title", "text", "label"]
df = pd.read_csv(f"{_DATA_PATH}/WELFake/WELFake_Dataset.csv")
df = df.dropna(subset=[columns[0]])

df.rename(
    columns={"title": "statement", "label": "verdict", "text": "content"}, inplace=True
)

del df["Unnamed: 0"]

df["verdict"] = df["verdict"].astype(str)  # Convert verdict to string type
df.loc[df["verdict"] == "1", "verdict"] = "true"
df.loc[df["verdict"] == "0", "verdict"] = "false"
df["verdict"] = df["verdict"].str.lower()  # Normalize verdicts

df.loc[df["statement"].notnull(), "statement"] = df[
    "statement"
].str.strip()  # Strip useless characters

df.to_csv(f"{_OUTPUT_PATH}/WELFake_Dataset.csv", index=False)
print("WELFake dataset processed and saved to WELFake_Dataset.csv")
