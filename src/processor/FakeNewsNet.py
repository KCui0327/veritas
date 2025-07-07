"""
Description: This script processes the FakeNewsNet dataset by reading multiple CSV files,
assigning verdicts based on the filename, and saving the combined dataset to a new CSV file.

The final DataFrame is saved to a new CSV file.

Author: Kenny Cui
Date: July 6, 2025
"""

import pandas as pd
import glob

_DATA_PATH = "../../data"
_OUTPUT_PATH = "../dataset/output"

dataset_files = glob.glob(f"{_DATA_PATH}/FakeNewsNet/*.csv")
dataset = []
for file in dataset_files:
    print(f"Processing {file}")
    df = pd.read_csv(file, header=0)
    if "real" in file:
        df['verdict'] = 'true'
    else:
        df['verdict'] = 'false'
    df = df.drop(columns=['id'])
    dataset.append(df)

df = pd.concat(dataset, ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True) # Shuffle the DataFrame
df = df.rename(columns={"title" : "statement", "text" : "content"})
df.loc[df['statement'].notnull(), 'statement'] = df['statement'].str.strip()

df.to_csv(f"{_OUTPUT_PATH}/FakeNewsNet.csv", index=False)
print("FakeNewsNet dataset processed and saved to FakeNewsNet.csv")
