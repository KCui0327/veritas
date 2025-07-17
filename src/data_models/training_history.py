import json
from dataclasses import dataclass, field
from typing import List

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from src.data_models.evaluation_metric import EvaluationMetric


@dataclass
class TrainingHistory:
    """Class to track training history for visualization."""

    epoch_nums: List[int] = field(default_factory=list)
    train_metrics: List[EvaluationMetric] = field(default_factory=list)
    val_metrics: List[EvaluationMetric] = field(default_factory=list)

    def add_epoch(
        self,
        epoch_num: int,
        train_metric: EvaluationMetric,
        val_metric: EvaluationMetric,
    ):
        self.epoch_nums.append(epoch_num)
        self.train_metrics.append(train_metric)
        self.val_metrics.append(val_metric)

    def plot_training_curves(self, save_path: str):
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

    def plot_confusion_matrix(self, save_path: str):
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
