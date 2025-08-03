import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from src.data_models.evaluation_metric import EvaluationMetric
from src.data_models.training_history import TrainingHistory
from src.util.logger import logger


def visualize_training_history(history: TrainingHistory, save_path: str):
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))

    train_epochs = [int(k) for k in history.train_metrics.keys()]
    val_epochs = [int(k) for k in history.val_metrics.keys()]
    axes[0, 0].set_xticks(train_epochs[::10])
    axes[0, 1].set_xticks(train_epochs[::10])
    axes[1, 0].set_xticks(train_epochs[::10])
    axes[1, 1].set_xticks(train_epochs[::10])
    axes[2, 0].set_xticks(train_epochs[::10])

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
    pass


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
    args = parser.parse_args()

    history_path = Path(args.history_path)
    output_path = args.output_path

    history_file_name = history_path.stem
    if not output_path:
        output_path = f"visualizations/{history_file_name}_history.png"

    with open(history_path, "r") as f:
        history = TrainingHistory(**json.load(f))

    visualize_training_history(history, save_path=output_path)


if __name__ == "__main__":
    main()
