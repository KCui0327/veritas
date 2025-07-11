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
        self.learning_rates = []
        self.epoch_times = []

    def add_epoch(
        self,
        train_loss: float,
        val_loss: Optional[float] = None,
        train_accuracy: Optional[float] = None,
        val_accuracy: Optional[float] = None,
        lr: Optional[float] = None,
        epoch_time: Optional[float] = None,
    ):
        """Add epoch results to history."""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss if val_loss is not None else float("inf"))
        self.train_accuracies.append(train_accuracy)
        self.val_accuracies.append(val_accuracy)
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
        }

        with open(filepath, "w") as f:
            json.dump(history_dict, f, indent=2)

    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

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
        axes[1, 0].plot(self.train_accuracies, label="Train Accuracy")
        if any(acc != 0.0 for acc in self.val_accuracies):
            axes[1, 0].plot(self.val_accuracies, label="Val Accuracy")
        axes[1, 0].set_title("Training and Validation Accuracy")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Accuracy")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Learning rate
        if self.learning_rates:
            axes[0, 1].plot(self.learning_rates)
            axes[0, 1].set_title("Learning Rate")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("Learning Rate")
            axes[0, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

        plt.close()
