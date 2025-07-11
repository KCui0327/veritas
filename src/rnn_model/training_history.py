import json
from typing import Optional

import matplotlib.pyplot as plt


class TrainingHistory:
    """Class to track training history for visualization."""

    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_precisions = []
        self.val_precisions = []
        self.train_recalls = []
        self.val_recalls = []
        self.train_f1_scores = []
        self.val_f1_scores = []
        self.learning_rates = []
        self.epoch_times = []

    def add_epoch(
        self,
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
    ):
        """Add epoch results to history."""
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
        if lr is not None:
            self.learning_rates.append(lr)

        if epoch_time is not None:
            self.epoch_times.append(epoch_time)

    def save(self, filepath: str):
        """Save training history to JSON file."""
        history_dict = {
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
        }

        with open(filepath, "w") as f:
            json.dump(history_dict, f, indent=2)

    def load(self, filepath: str):
        """Load training history from JSON file."""
        with open(filepath, "r") as f:
            history_dict = json.load(f)
        self.train_losses = history_dict["train_losses"]
        self.val_losses = history_dict["val_losses"]
        self.train_accuracies = history_dict["train_accuracies"]
        self.val_accuracies = history_dict["val_accuracies"]
        self.learning_rates = history_dict["learning_rates"]
        self.epoch_times = history_dict["epoch_times"]

        # Load new metrics if available (for backward compatibility)
        self.train_precisions = history_dict.get("train_precisions", [])
        self.val_precisions = history_dict.get("val_precisions", [])
        self.train_recalls = history_dict.get("train_recalls", [])
        self.val_recalls = history_dict.get("val_recalls", [])
        self.train_f1_scores = history_dict.get("train_f1_scores", [])
        self.val_f1_scores = history_dict.get("val_f1_scores", [])

    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves."""
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))

        # Loss curves
        axes[0, 0].plot(self.train_losses, label="Train Loss")
        if any(loss != float("inf") for loss in self.val_losses):
            axes[0, 0].plot(self.val_losses, label="Val Loss")
        axes[0, 0].set_title("Training and Validation Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Accuracy curves
        axes[0, 1].plot(self.train_accuracies, label="Train Accuracy")
        if any(acc != 0.0 for acc in self.val_accuracies):
            axes[0, 1].plot(self.val_accuracies, label="Val Accuracy")
        axes[0, 1].set_title("Training and Validation Accuracy")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Precision curves
        if self.train_precisions:
            axes[1, 0].plot(self.train_precisions, label="Train Precision")
            if any(p is not None for p in self.val_precisions):
                val_precisions_clean = [p for p in self.val_precisions if p is not None]
                val_indices = [
                    i for i, p in enumerate(self.val_precisions) if p is not None
                ]
                axes[1, 0].plot(
                    val_indices, val_precisions_clean, label="Val Precision"
                )
            axes[1, 0].set_title("Training and Validation Precision")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Precision")
            axes[1, 0].legend()
            axes[1, 0].grid(True)

        # Recall curves
        if self.train_recalls:
            axes[1, 1].plot(self.train_recalls, label="Train Recall")
            if any(r is not None for r in self.val_recalls):
                val_recalls_clean = [r for r in self.val_recalls if r is not None]
                val_indices = [
                    i for i, r in enumerate(self.val_recalls) if r is not None
                ]
                axes[1, 1].plot(val_indices, val_recalls_clean, label="Val Recall")
            axes[1, 1].set_title("Training and Validation Recall")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Recall")
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        # F1 Score curves
        if self.train_f1_scores:
            axes[2, 0].plot(self.train_f1_scores, label="Train F1 Score")
            if any(f1 is not None for f1 in self.val_f1_scores):
                val_f1_clean = [f1 for f1 in self.val_f1_scores if f1 is not None]
                val_indices = [
                    i for i, f1 in enumerate(self.val_f1_scores) if f1 is not None
                ]
                axes[2, 0].plot(val_indices, val_f1_clean, label="Val F1 Score")
            axes[2, 0].set_title("Training and Validation F1 Score")
            axes[2, 0].set_xlabel("Epoch")
            axes[2, 0].set_ylabel("F1 Score")
            axes[2, 0].legend()
            axes[2, 0].grid(True)

        # Learning rate
        if self.learning_rates:
            axes[2, 1].plot(self.learning_rates)
            axes[2, 1].set_title("Learning Rate")
            axes[2, 1].set_xlabel("Epoch")
            axes[2, 1].set_ylabel("Learning Rate")
            axes[2, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

        plt.close()
