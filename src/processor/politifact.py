"""
Description: This script processes the Politifact fact-checking dataset,
extracting relevant fields, and filtering out certain verdicts.

The final DataFrame is saved to a new CSV file.

Author: Kenny Cui
Date: July 6, 2025
"""

import json
import pandas as pd

HEADER = [
    "statement",
    "verdict",
    "statement_originator",
    "statement_date",
    "statement_url",
    "factchecker",
    "factcheck_date",
    "factcheck_url",
]

df = pd.DataFrame(columns=HEADER)
verdict_discard = {"mostly-false", "half-true", "mostly-true"}

_DATA_PATH = "../../data"
_OUTPUT_PATH = "../dataset/output"

with open(
    f"{_DATA_PATH}/Politifact/politifact_factcheck_data.json", "r", encoding="utf-8"
) as file:
    for line in file:
        data_point = json.loads(line)

        verdict = data_point.get("verdict", "")
        if verdict == "" or verdict in verdict_discard:
            continue
        if verdict == "pants on fire":
            verdict = "false"

        statement = data_point.get("statement", "")
        if statement == "":
            continue

        statement_originator = data_point.get("statement_originator", "")
        statement_date = data_point.get("statement_date", "")
        statement_url = data_point.get("statement_url", "")
        factchecker = data_point.get("factchecker", "")
        factcheck_date = data_point.get("factcheck_date", "")
        factcheck_url = data_point.get("factcheck_url", "")
        data = [
            statement_originator,
            statement_date,
            statement_url,
            factchecker,
            factcheck_date,
            factcheck_url,
        ]

        df.loc[len(df)] = [statement, verdict] + data

df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the DataFrame
df.loc[df["statement"].notnull(), "statement"] = df["statement"].str.strip()
df.loc[df["verdict"] == "pants-fire", "verdict"] = (
    "false"  # Merge "pants-fire" into "false"
)
df["verdict"] = df["verdict"].str.lower()  # Normalize verdicts

# Removing unnecessary columns
del df["factchecker"]
del df["factcheck_date"]
del df["factcheck_url"]
del df["statement_originator"]
del df["statement_url"]

df.rename(columns={"statement_date": "date"}, inplace=True)
df.to_csv(f"{_OUTPUT_PATH}/politifact.csv", index=False, encoding="utf-8")
print("Politifact dataset processed and saved to politifact.csv")
