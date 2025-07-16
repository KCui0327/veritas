import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix


def main():
    y_true, y_pred, model, file_name = pickle.load(open("model.pkl", "rb"))

    # Set up the plotting style
    plt.style.use("default")
    sns.set_palette("husl")

    # Create a figure with subplots (2 rows, 2 columns instead of 2x3)
    fig = plt.figure(figsize=(16, 12))

    # 1. Confusion Matrix
    plt.subplot(2, 3, 1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["True News", "Fake News"],
        yticklabels=["True News", "Fake News"],
    )
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    # 2. Performance Metrics Bar Chart
    plt.subplot(2, 3, 2)
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    values = [accuracy_score(y_true, y_pred), precision, recall, f1_score]
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]

    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.title("Model Performance Metrics", fontsize=14, fontweight="bold")
    plt.ylabel("Score")
    plt.ylim(0, 1)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 3. Class Balance Check
    plt.subplot(2, 3, 3)
    class_counts = np.bincount(y_true)
    total_samples = len(y_true)
    labels = [
        f"True News\n({class_counts[0]:,} samples)",
        f"Fake News\n({class_counts[1]:,} samples)",
    ]
    colors = ["#2E86AB", "#A23B72"]

    plt.pie(
        class_counts, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90
    )
    plt.title(
        f"Class Distribution\n(Total: {total_samples:,} samples)",
        fontsize=14,
        fontweight="bold",
    )

    # Create visualizations directory if it doesn't exist
    viz_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "visualizations"
    )
    os.makedirs(viz_dir, exist_ok=True)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(
        os.path.join(viz_dir, file_name),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()  # Close the figure to free memory

    # Print detailed metrics
    print(f"\nDetailed Performance Metrics:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")
    print(f"\nVisualizations saved as '{os.path.join(viz_dir, file_name)}'")


if __name__ == "__main__":
    main()
