"""
Description: Visualize the Veritas dataset.

Author: Kenny Cui
Date: July 6, 2025

Credits to: https://www.geeksforgeeks.org/python/plot-multiple-plots-in-matplotlib/
"""

import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

df = pd.read_csv("veritas_dataset.csv")

# Create a 2x2 plot grid
figure, axis = plt.subplots(2, 2)

# Total number of statements
total_statements = len(df)
axis[0, 0].set_title("Total Statements")
print(f"Total Statements: {total_statements}")
axis[0, 0].bar(["Total Statements"], [total_statements], color="blue")

# Length of each statement
axis[0, 1].set_title("Length of Each Statement")
statement_length = defaultdict(int)
for _, row in df.iterrows():
    length = len(row["statement"].split())
    statement_length[length] += 1
print(statement_length)
axis[0, 1].bar(statement_length.keys(), statement_length.values(), color="orange")

# Truth count vs. False count
axis[1, 0].set_title("Truth Count vs. False Count")
verdicts = df["verdict"].value_counts()
print(verdicts)
axis[1, 0].bar(["False", "True"], verdicts.values, color="orange")

# Types of news sources
axis[1, 1].set_title("Types of News Sources")
type_sources = df["subject"].value_counts()

# Merge scattered types into "Others" if they have less than 500 occurrences
others = type_sources[type_sources <= 500].sum()
type_sources = type_sources[type_sources > 500]._append(pd.Series({"Others": others}))

print(type_sources)
axis[1, 1].bar(type_sources.index, type_sources.values, color="green")

plt.show()
