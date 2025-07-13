import json
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns


class TrainingHistory:
    """Class to track training history for visualization."""

    def __init__(self):
        self.epochs = []
        self.epoch_times = []

        self.train_losses = []
        self.val_losses = []

        self.train_accuracies = []
        self.train_precisions = []
        self.train_recalls = []
        self.train_f1_scores = []
        self.train_confusion_matrices = []

        self.val_accuracies = []
        self.val_precisions = []
        self.val_recalls = []
        self.val_f1_scores = []
        self.val_confusion_matrices = []

        self.learning_rates = []

    def add_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        train_accuracy: Optional[float] = None,
        val_accuracy: Optional[float] = None,
        train_precision: Optional[float] = None,
        val_precision: Optional[float] = None,
        train_recall: Optional[float] = None,
        val_recall: Optional[float] = None,
        train_f1: Optional[float] = None,
        val_f1: Optional[float] = None,
        lr: Optional[float] = None,
        epoch_time: Optional[float] = None,
        train_confusion_matrix: Optional[np.ndarray] = None,
        val_confusion_matrix: Optional[np.ndarray] = None,
    ):
        """Add epoch results to history."""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss if val_loss is not None else float("inf"))
        self.train_accuracies.append(train_accuracy)
        self.val_accuracies.append(val_accuracy)
        self.train_precisions.append(train_precision)
        self.val_precisions.append(val_precision)
        self.train_recalls.append(train_recall)
        self.val_recalls.append(val_recall)
        self.train_f1_scores.append(train_f1)
        self.val_f1_scores.append(val_f1)
        self.train_confusion_matrices.append(train_confusion_matrix)
        self.val_confusion_matrices.append(val_confusion_matrix)

        if lr is not None:
            self.learning_rates.append(lr)

        self.epoch_times.append(epoch_time)

    def save(self, filepath: str):
        """Save training history to JSON file."""
        history_dict = {
            "epochs": self.epochs,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates,
            "epoch_times": self.epoch_times,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies,
            "train_precisions": self.train_precisions,
            "val_precisions": self.val_precisions,
            "train_recalls": self.train_recalls,
            "val_recalls": self.val_recalls,
            "train_f1_scores": self.train_f1_scores,
            "val_f1_scores": self.val_f1_scores,
            "train_confusion_matrices": self.train_confusion_matrices,
            "val_confusion_matrices": self.val_confusion_matrices,
        }

        with open(filepath, "w") as f:
            json.dump(history_dict, f, indent=2)

    def load(self, filepath: str):
        """Load training history from JSON file."""
        with open(filepath, "r") as f:
            history_dict = json.load(f)
        self.epochs = history_dict["epochs"]
        self.train_losses = history_dict["train_losses"]
        self.val_losses = history_dict["val_losses"]
        self.train_accuracies = history_dict["train_accuracies"]
        self.val_accuracies = history_dict["val_accuracies"]
        self.learning_rates = history_dict["learning_rates"]
        self.epoch_times = history_dict["epoch_times"]

        self.train_precisions = history_dict["train_precisions"]
        self.val_precisions = history_dict["val_precisions"]
        self.train_recalls = history_dict["train_recalls"]
        self.val_recalls = history_dict["val_recalls"]
        self.train_f1_scores = history_dict["train_f1_scores"]
        self.val_f1_scores = history_dict["val_f1_scores"]
        # self.train_confusion_matrices = history_dict["train_confusion_matrices"]
        # self.val_confusion_matrices = history_dict["val_confusion_matrices"]

    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves."""
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))

        for ax in axes.flat:
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        # Loss curves
        axes[0, 0].plot(self.epochs, self.train_losses, label="Train Loss")
        axes[0, 0].plot(self.epochs, self.val_losses, label="Val Loss")
        axes[0, 0].set_title("Training and Validation Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Accuracy curves
        axes[0, 1].plot(self.epochs, self.train_accuracies, label="Train Accuracy")
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].plot(self.epochs, self.val_accuracies, label="Val Accuracy")
        axes[0, 1].set_title("Training and Validation Accuracy")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Precision curves
        axes[1, 0].plot(self.epochs, self.train_precisions, label="Train Precision")
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].plot(self.epochs, self.val_precisions, label="Val Precision")
        axes[1, 0].set_title("Training and Validation Precision")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Precision")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Recall curves
        axes[1, 1].plot(self.epochs, self.train_recalls, label="Train Recall")
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].plot(self.epochs, self.val_recalls, label="Val Recall")
        axes[1, 1].set_title("Training and Validation Recall")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Recall")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        # F1 Score curves
        axes[2, 0].plot(self.epochs, self.train_f1_scores, label="Train F1 Score")
        axes[2, 0].set_ylim(0, 1)
        axes[2, 0].plot(self.epochs, self.val_f1_scores, label="Val F1 Score")
        axes[2, 0].set_title("Training and Validation F1 Score")
        axes[2, 0].set_xlabel("Epoch")
        axes[2, 0].set_ylabel("F1 Score")
        axes[2, 0].legend()
        axes[2, 0].grid(True)

        metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        values = [
            self.val_accuracies[-1],
            self.val_precisions[-1],
            self.val_recalls[-1],
            self.val_f1_scores[-1],
        ]
        colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]

        bars = axes[2, 1].bar(metrics, values, color=colors, alpha=0.7)
        axes[2, 1].set_title(
            "Model Performance Metrics (Validation)",
            fontsize=14,
            fontweight="bold",
        )
        axes[2, 1].set_ylabel("Score", fontsize=12)
        axes[2, 1].set_ylim(0, 1)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[2, 1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

        plt.close()

    def plot_confusion_matrix(self, save_path: Optional[str] = None):
        """Plot confusion matrix."""
        cm = self.val_confusion_matrices[-1]
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

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

        plt.close()
