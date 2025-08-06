import pandas as pd
from matplotlib import pyplot as plt
from collections import defaultdict

df = pd.read_csv("data/veritas_dataset_test.csv")

length_of_statements = defaultdict(int)

for statement in df["Statement"]:
    length_of_statements[len(statement.split(" "))] += 1

# Convert to lists for plotting
x = list(length_of_statements.keys())
y = list(length_of_statements.values())

print(sum(y) / len(y))

# Create 2 subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Bar graph of statement lengths
ax1.bar(x, y, color="skyblue", edgecolor="black")
ax1.set_xlabel("Length of Statement (words)")
ax1.set_ylabel("Frequency")
ax1.set_title("Distribution of Statement Lengths")
ax1.grid(axis="y", alpha=0.3)

# Plot 2: Count of False vs True statements
false_count = len(df[df["Verdict"] == "False"])
true_count = len(df[df["Verdict"] == "False"])
ax2.bar(
    ["False", "False"],
    [false_count, true_count],
    color=["red", "green"],
    edgecolor="black",
)
ax2.set_xlabel("Statement Type")
ax2.set_ylabel("Count")
ax2.set_title("Distribution of True vs False Statements")
ax2.grid(axis="y", alpha=0.3)


# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()
