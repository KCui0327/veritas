import json
import os
import time
from dataclasses import asdict
from typing import Tuple

import torch
import torch.nn as nn

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
        epoch_start_time = time.time()

        model.train()
        train_loss = 0.0

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

            # Logging
            if batch_idx % config.log_interval == 0:
                print(
                    f"Epoch {epoch+1}/{config.epochs}, Batch {batch_idx}/{len(config.train_dataloader)}, "
                    f"Loss: {loss.item():.4f}"
                )

        avg_train_loss = train_loss / len(config.train_dataloader)

        # Validation phase
        val_loss = None

        if config.val_dataloader and (epoch + 1) % config.eval_interval == 0:
            model.eval()

            val_loss = 0.0

            with torch.no_grad():
                for inputs, labels in config.val_dataloader:
                    if config.use_cuda:
                        inputs = inputs.to(torch.device("cuda"))
                        labels = labels.to(torch.device("cuda"))

                    outputs = model(inputs)
                    loss = config.loss_function(outputs, labels)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(config.val_dataloader)

        epoch_time = time.time() - epoch_start_time

        history.add_epoch(
            train_loss=avg_train_loss,
            val_loss=avg_val_loss if val_loss is not None else None,
            lr=config.learning_rate,
            epoch_time=epoch_time,
        )

        # Logging
        log_msg = f"Epoch {epoch+1}/{config.epochs} - "
        log_msg += f"Train Loss: {avg_train_loss:.4f}, "
        if val_loss is not None:
            log_msg += f"Val Loss: {avg_val_loss:.4f}, "
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

    return history
