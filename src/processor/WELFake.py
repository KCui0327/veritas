"""
Description: This script processes the WELFake dataset by reading a CSV file,

The final DataFrame is saved to a new CSV file.

Author: Kenny Cui
Date: July 6, 2025
"""

import pandas as pd

_DATA_PATH = "../../data"
_OUTPUT_PATH = "../dataset/output"

columns = ["title", "text", "label"]
df = pd.read_csv(f"{_DATA_PATH}/WELFake/WELFake_Dataset.csv")
df = df.dropna(subset=[columns[0]])

df = df.rename(columns={"title": "statement", "label": "verdict"})
df.loc[df["verdict"] == 1, "verdict"] = 'true'
df.loc[df["verdict"] == 0, "verdict"] = 'false'
df.loc[df['statement'].notnull(), 'statement'] = df['statement'].str.strip()

df.to_csv(f"{_OUTPUT_PATH}/WELFake_Dataset.csv", index=False)
print("WELFake dataset processed and saved to WELFake_Dataset.csv")
