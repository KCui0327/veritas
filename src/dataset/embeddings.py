from transformers import BertTokenizer, BertModel
import torch
import os
import pandas as pd
from pathlib import Path
import glob
import numpy as np

DATASET_NAME = "veritas_dataset.csv"
PROJECT_DIR = Path(__file__).resolve().parents[3]
DATA_PATH = os.path.join(PROJECT_DIR, "data")

dataset = glob.glob(f"{DATA_PATH}/{DATASET_NAME}")
dataset = pd.read_csv(dataset[0])

len_dataset = len(dataset)
print(f"Dataset loaded with {len_dataset} rows.")

embeddings = []
labels = []

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

for index, row in dataset.iterrows():
    phrase = row['statement']
    tokens = tokenizer(phrase, return_tensors="pt")

    if index % 100 == 0:
        print(f"Processing row {index}: {phrase}")

    with torch.no_grad():
        outputs = model(**tokens)

    tokens_embedding = outputs.last_hidden_state
    tokens_embedding = tokens_embedding[0, 1:-1]  # Exclude [CLS] and [SEP] tokens


    embeddings.append(tokens_embedding)
    labels.append(str(row['verdict']))

np.savez_compressed("veritas_dataset.npz", embeddings=np.array(embeddings, dtype=object),
         labels=np.array(labels, dtype=object))

print(f"Embeddings and labels saved to veritas_dataset.npz")
