import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.data_models.evaluation_metric import EvaluationMetric
from src.data_models.training_history import TrainingHistory
from src.util.logger import logger


def visualize_training_history(history: TrainingHistory, save_path: str):
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))

    train_epochs = [int(k) for k in history.train_metrics.keys()]
    val_epochs = [int(k) for k in history.val_metrics.keys()]
    axes[0, 0].set_xticks(train_epochs[::5])
    axes[0, 1].set_xticks(train_epochs[::5])
    axes[1, 0].set_xticks(train_epochs[::5])
    axes[1, 1].set_xticks(train_epochs[::5])
    axes[2, 0].set_xticks(train_epochs[::5])

    # plot training and validation loss history
    axes[0, 0].plot(
        train_epochs,
        [metric["avg_loss"] for metric in history.train_metrics.values()],
        label="Training Loss",
    )
    axes[0, 0].plot(
        val_epochs,
        [metric["avg_loss"] for metric in history.val_metrics.values()],
        label="Validation Loss",
    )
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].set_title("Training and Validation Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")

    # plot training and validation accuracy history
    axes[0, 1].plot(
        train_epochs,
        [metric["accuracy"] for metric in history.train_metrics.values()],
        label="Training Accuracy",
    )
    axes[0, 1].plot(
        val_epochs,
        [metric["accuracy"] for metric in history.val_metrics.values()],
        label="Validation Accuracy",
    )
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].set_title("Training and Validation Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")

    # plot training and validation precision history
    axes[1, 0].plot(
        train_epochs,
        [metric["precision"] for metric in history.train_metrics.values()],
        label="Training Precision",
    )
    axes[1, 0].plot(
        val_epochs,
        [metric["precision"] for metric in history.val_metrics.values()],
        label="Validation Precision",
    )
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].set_title("Training and Validation Precision")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Precision")
    plt.savefig(save_path)
    plt.close()

    # plot training and validation recall history
    axes[1, 1].plot(
        train_epochs,
        [metric["recall"] for metric in history.train_metrics.values()],
        label="Training Recall",
    )
    axes[1, 1].plot(
        val_epochs,
        [metric["recall"] for metric in history.val_metrics.values()],
        label="Validation Recall",
    )
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].set_title("Training and Validation Recall")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Recall")

    # plot training and validation f1 score history
    axes[2, 0].plot(
        train_epochs,
        [metric["f1_score"] for metric in history.train_metrics.values()],
        label="Training F1 Score",
    )
    axes[2, 0].plot(
        val_epochs,
        [metric["f1_score"] for metric in history.val_metrics.values()],
        label="Validation F1 Score",
    )
    axes[2, 0].legend()
    axes[2, 0].grid(True)
    axes[2, 0].set_title("Training and Validation F1 Score")
    axes[2, 0].set_xlabel("Epoch")
    axes[2, 0].set_ylabel("F1 Score")

    axes[2, 1].set_visible(False)

    logger.info(f"Saving training history to {save_path}")
    fig.savefig(save_path)


def visualize_evaluation_metric(metric: EvaluationMetric, save_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    metrics_values = [metric.accuracy, metric.precision, metric.recall, metric.f1_score]

    bars = axes[0].bar(
        metrics_names,
        metrics_values,
        color=["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"],
    )
    axes[0].set_title("Model Performance Metrics")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, 1.1)  # Add extra space at the top for labels
    axes[0].grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        # Position text inside the bar if it's too high, otherwise above
        if value > 0.9:  # If bar is very high
            text_y = bar.get_height() - 0.05  # Position inside the bar
            text_color = "white"
            va = "top"
        else:
            text_y = bar.get_height() + 0.01  # Position above the bar
            text_color = "black"
            va = "bottom"

        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            text_y,
            f"{value:.3f}",
            ha="center",
            va=va,
            color=text_color,
            fontweight="bold",
        )

    confusion_matrix = np.array(
        [
            [metric.num_true_negatives, metric.num_false_positives],
            [metric.num_false_negatives, metric.num_true_positives],
        ]
    )

    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Fake", "Predicted Real"],
        yticklabels=["Actual Fake", "Actual Real"],
        ax=axes[1],
    )
    axes[1].set_title("Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    # 3. Dataset Statistics - Distribution of true vs false labels
    dataset_labels = ["False/Fake News", "True/Real News"]
    dataset_counts = [metric.dataset_num_false_labels, metric.dataset_num_true_labels]

    colors = ["#2E86AB", "#A23B72"]

    # Create custom autopct function to show both percentage and raw count
    def autopct_func(pct):
        absolute = int(pct / 100.0 * sum(dataset_counts))
        return f"{pct:.1f}%\n({absolute:,})"

    wedges, texts, autotexts = axes[2].pie(
        dataset_counts,
        labels=dataset_labels,
        colors=colors,
        autopct=autopct_func,
        startangle=90,
    )
    axes[2].set_title("Dataset Label Distribution")

    logger.info(f"Saving evaluation metrics visualization to {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history-path",
        type=str,
        required=True,
        help="Path to the training history file",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=False,
        help="Path to the output file",
    )
    parser.add_argument(
        "--type",
        choices=["history", "evaluation"],
        required=True,
        help="Type of visualization to perform",
    )
    parser.add_argument(
        "--epoch",
        type=str,
        required=False,
        help="Epoch to visualize",
    )
    args = parser.parse_args()

    history_path = Path(args.history_path)
    output_path = args.output_path

    history_file_name = history_path.stem
    if not output_path:
        output_path = f"visualizations/{history_file_name}.png"

    if args.type == "history":

        with open(history_path, "r") as f:
            history = TrainingHistory(**json.load(f))
            visualize_training_history(history, save_path=output_path)

    elif args.type == "evaluation":
        if args.epoch:
            with open(history_path, "r") as f:
                history = TrainingHistory(**json.load(f))
                evaluation_metric = EvaluationMetric(**history.val_metrics[args.epoch])
                visualize_evaluation_metric(evaluation_metric, save_path=output_path)
        else:
            with open(history_path, "r") as f:
                evaluation_metric = EvaluationMetric(**json.load(f))
                visualize_evaluation_metric(evaluation_metric, save_path=output_path)


if __name__ == "__main__":
    main()
