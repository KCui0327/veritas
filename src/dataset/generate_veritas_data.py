import glob
import hashlib
import os
import runpy

import pandas as pd

# Get the directory where this script is located
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATASET_PATH = os.path.join(_CURRENT_DIR, "output")
_PROCESSOR_PATH = os.path.join(_CURRENT_DIR, "processor")
_OUTPUT_FILE = "data/veritas_dataset.csv"


def main():
    for file in os.listdir(_PROCESSOR_PATH):
        if file.endswith(".py"):
            print(f"Running processor: {file}")
            runpy.run_path(os.path.join(_PROCESSOR_PATH, file))

    dataset_files = glob.glob(f"{_DATASET_PATH}/*.csv")
    datasets = []
    for file in dataset_files:
        print(f"Processing {file}")
        df = pd.read_csv(file, header=0)
        datasets.append(df)

    df = pd.concat(datasets, ignore_index=True)
    # random state is seed
    df = df.sample(frac=1, random_state=21).reset_index(
        drop=True
    )  # Shuffle the DataFrame

    # Remove duplicates based on the 'statement' column through unique hashing
    df["id"] = df["statement"].apply(
        lambda x: hashlib.md5(x.encode("utf-8")).hexdigest()
    )

    df.drop_duplicates(subset=["id"], inplace=True, keep="first")

    df.to_csv(_OUTPUT_FILE, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
    print("Dataset processing complete. DataLoader created.")
