import json
import os
import time
from dataclasses import asdict
from typing import Tuple

import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score

from src.dataset import get_dataloaders
from src.rnn_model.model import FakeNewsDetector
from src.rnn_model.train import train_model
from src.rnn_model.training_config import TrainingConfig
from src.rnn_model.training_history import TrainingHistory


def train_model(
    model: nn.Module, config: TrainingConfig
) -> Tuple[nn.Module, TrainingHistory]:
    """
    Generic PyTorch training function.

    Args:
        config: TrainingConfig object containing all training parameters

    Returns:
        Tuple of (trained_model, training_history)
    """

    history = TrainingHistory()
    os.makedirs(config.save_dir, exist_ok=True)

    if config.use_cuda:
        print("Using CUDA")
        model.to(torch.device("cuda"))

    print(f"Starting training for {config.epochs} epochs...")
    print(f"Training samples: {len(config.train_dataloader)}")
    if config.val_dataloader:
        print(f"Validation samples: {len(config.val_dataloader)}")

    for epoch in range(config.epochs):
        print(f"Epoch {epoch+1}/{config.epochs}")
        epoch_start_time = time.time()

        model.train()
        train_loss = 0.0

        train_correct = 0
        train_total = 0
        all_train_preds = []
        all_train_labels = []

        for batch_idx, (inputs, labels) in enumerate(config.train_dataloader):
            if config.use_cuda:
                inputs = inputs.to(torch.device("cuda"))
                labels = labels.to(torch.device("cuda"))

            config.optimizer.zero_grad()

            outputs = model(inputs)
            loss = config.loss_function(outputs, labels)
            loss.backward()
            config.optimizer.step()

            train_loss += loss.item()

            with torch.no_grad():
                preds = (outputs >= 0.5).float()
                train_correct += (preds == labels).sum().item()
                train_total += labels.numel()

                # Store predictions and labels for metric calculations
                all_train_preds.extend(preds.cpu().numpy().flatten())
                all_train_labels.extend(labels.cpu().numpy().flatten())

            if batch_idx % config.log_interval == 0:
                batch_acc = (train_correct / train_total) if train_total > 0 else 0.0
                print(
                    f"Epoch {epoch+1}/{config.epochs}, Batch {batch_idx}/{len(config.train_dataloader)}, "
                    f"Loss: {loss.item():.4f}, Acc: {batch_acc:.4f}"
                )

        avg_train_loss = train_loss / len(config.train_dataloader)
        avg_train_acc = train_correct / train_total if train_total > 0 else 0.0

        # Calculate training metrics
        train_precision = precision_score(
            all_train_labels, all_train_preds, average="binary", zero_division=0
        )
        train_recall = recall_score(
            all_train_labels, all_train_preds, average="binary", zero_division=0
        )
        train_f1 = f1_score(
            all_train_labels, all_train_preds, average="binary", zero_division=0
        )

        # Validation phase
        val_loss = None
        val_precision = None
        val_recall = None
        val_f1 = None

        if config.val_dataloader and (epoch + 1) % config.eval_interval == 0:
            model.eval()

            val_loss = 0.0
            val_correct = 0
            val_total = 0
            all_val_preds = []
            all_val_labels = []

            with torch.no_grad():
                for inputs, labels in config.val_dataloader:
                    if config.use_cuda:
                        inputs = inputs.to(torch.device("cuda"))
                        labels = labels.to(torch.device("cuda"))

                    outputs = model(inputs)
                    loss = config.loss_function(outputs, labels)
                    val_loss += loss.item()

                    preds = (outputs >= 0.5).float()
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.numel()

                    # Store predictions and labels for metric calculations
                    all_val_preds.extend(preds.cpu().numpy().flatten())
                    all_val_labels.extend(labels.cpu().numpy().flatten())

            avg_val_loss = val_loss / len(config.val_dataloader)
            avg_val_acc = val_correct / val_total if val_total > 0 else 0.0

            # Calculate validation metrics
            val_precision = precision_score(
                all_val_labels, all_val_preds, average="binary", zero_division=0
            )
            val_recall = recall_score(
                all_val_labels, all_val_preds, average="binary", zero_division=0
            )
            val_f1 = f1_score(
                all_val_labels, all_val_preds, average="binary", zero_division=0
            )

        epoch_time = time.time() - epoch_start_time

        history.add_epoch(
            epoch=epoch,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss if val_loss is not None else None,
            lr=config.learning_rate,
            epoch_time=epoch_time,
            train_accuracy=avg_train_acc,
            val_accuracy=avg_val_acc,
            train_precision=train_precision,
            val_precision=val_precision,
            train_recall=train_recall,
            val_recall=val_recall,
            train_f1=train_f1,
            val_f1=val_f1,
        )

        # Logging
        log_msg = f"Epoch {epoch+1}/{config.epochs} - "
        log_msg += f"Train Loss: {avg_train_loss:.4f}, "
        log_msg += f"Train Acc: {avg_train_acc:.4f}, "
        log_msg += f"Train Prec: {train_precision:.4f}, "
        log_msg += f"Train Rec: {train_recall:.4f}, "
        log_msg += f"Train F1: {train_f1:.4f}, "
        if val_loss is not None:
            log_msg += f"Val Loss: {avg_val_loss:.4f}, "
            log_msg += f"Val Acc: {avg_val_acc:.4f}, "
            log_msg += f"Val Prec: {val_precision:.4f}, "
            log_msg += f"Val Rec: {val_recall:.4f}, "
            log_msg += f"Val F1: {val_f1:.4f}, "
        log_msg += f"LR: {config.learning_rate:.6f}, Time: {epoch_time:.2f}s"
        print(log_msg)

        model_name = f"{model.name}_{config.get_config_unique_name()}_{epoch}.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": config.optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss if val_loss is not None else None,
                "config": asdict(config),
            },
            os.path.join(config.save_dir, model_name),
        )

    print("Training completed!")

    history_name = f"{model.name}_{config.get_config_unique_name()}.json"
    history.save(os.path.join(config.save_dir, history_name))

    return model, history


def load_trained_model(
    model: nn.Module, checkpoint_path: str
) -> Tuple[nn.Module, TrainingConfig]:
    """
    Load a trained model from checkpoint.

    Args:
        model: The model architecture
        checkpoint_path: Path to the checkpoint file

    Returns:
        Tuple of (loaded_model, training_config)
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint["config"]


def load_training_history(history_path: str) -> TrainingHistory:
    """
    Load training history from JSON file.

    Args:
        history_path: Path to the history JSON file

    Returns:
        TrainingHistory object
    """
    history = TrainingHistory()

    with open(history_path, "r") as f:
        history_dict = json.load(f)

    history.train_losses = history_dict["train_losses"]
    history.val_losses = history_dict["val_losses"]
    history.learning_rates = history_dict["learning_rates"]
    history.epoch_times = history_dict["epoch_times"]
    history.train_accuracies = history_dict.get("train_accuracies", [])
    history.val_accuracies = history_dict.get("val_accuracies", [])
    history.train_precisions = history_dict.get("train_precisions", [])
    history.val_precisions = history_dict.get("val_precisions", [])
    history.train_recalls = history_dict.get("train_recalls", [])
    history.val_recalls = history_dict.get("val_recalls", [])
    history.train_f1_scores = history_dict.get("train_f1_scores", [])
    history.val_f1_scores = history_dict.get("val_f1_scores", [])

    return history


def main():
    # Hyperparameters
    vocab_size = 20000
    embed_dim = 128
    hidden_dim = 128
    num_layers = 2
    dropout_rate = 0.5
    bidirectional = True
    output_dim = 1
    lr = 1e-2

    print("Creating model")
    model = FakeNewsDetector(
        vocab_size,
        embed_dim,
        hidden_dim,
        num_layers,
        dropout_rate,
        bidirectional,
        output_dim,
    )

    print("Getting dataloaders")
    train_dataloader, val_dataloader = get_dataloaders(batch_size=64, max_records=1000)

    print("Creating optimizer")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    training_config = TrainingConfig(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        save_dir="checkpoints",
        log_interval=16,
        eval_interval=1,
        epochs=20,
        optimizer=optimizer,
        loss_function=nn.BCELoss(),
        use_cuda=torch.cuda.is_available(),
    )

    train_model(model, training_config)


if __name__ == "__main__":
    main()
