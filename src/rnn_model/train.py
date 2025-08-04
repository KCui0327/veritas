import argparse
import json
import os
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn

from src.data_models.evaluation_metric import EvaluationMetric
from src.data_models.training_config import TrainingConfig
from src.data_models.training_history import TrainingHistory
from src.dataset.get_data_loaders import get_dataloaders
from src.rnn_model.evaluate import evaluate_model
from src.rnn_model.model import FakeNewsDetector
from src.util.logger import logger


_DATASET_PTH = "data/veritas_dataset.csv"


def train_model(model: nn.Module, config: TrainingConfig) -> TrainingHistory:
    history = TrainingHistory()

    if config.use_cuda:
        logger.info("Using CUDA")
        logger.info("PyTorch version:", torch.__version__)
        logger.info("CUDA available:", torch.cuda.is_available())
        logger.info("CUDA device count:", torch.cuda.device_count())
        logger.info(f"Current device: {torch.cuda.current_device()}")
        logger.info(f"Device name: {torch.cuda.get_device_name(0)}")

        model.to(torch.device("cuda"))

    logger.info(f"Starting training for {config.epochs} epochs...")
    logger.info(f"Training samples: {len(config.train_dataloader.dataset)}")
    logger.info(f"Validation samples: {len(config.val_dataloader.dataset)}")

    for epoch in range(config.epochs):
        logger.info(f"Epoch {epoch+1}/{config.epochs}")

        epoch_start_time = time.time()
        model.train()
        train_metric = EvaluationMetric(
            batch_size=config.train_dataloader.batch_size,
            num_model_parameters=sum(p.numel() for p in model.parameters()),
        )
        all_train_labels = []
        all_train_preds = []

        for _, (inputs, labels) in enumerate(config.train_dataloader):
            if config.use_cuda:
                inputs = inputs.to(torch.device("cuda"))
                labels = labels.to(torch.device("cuda"))

            config.optimizer.zero_grad()

            outputs = model(inputs)
            preds = (outputs >= 0.5).float()

            loss = config.loss_function(outputs, labels)
            loss.backward()
            config.optimizer.step()

            train_metric.total_loss += loss.item()
            train_metric.dataset_size += inputs.size(0)

            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(preds.cpu().numpy())

        epoch_time = time.time() - epoch_start_time
        train_metric.time_taken = epoch_time

        train_metric.dataset_num_true_labels = int(sum(all_train_labels))
        train_metric.dataset_num_false_labels = int(
            len(all_train_labels) - train_metric.dataset_num_true_labels
        )

        all_train_labels = np.array(all_train_labels)
        all_train_preds = np.array(all_train_preds)

        train_metric.num_true_positives = int(
            sum((all_train_labels == 1) & (all_train_preds == 1))
        )
        train_metric.num_false_positives = int(
            sum((all_train_labels == 0) & (all_train_preds == 1))
        )
        train_metric.num_true_negatives = int(
            sum((all_train_labels == 0) & (all_train_preds == 0))
        )
        train_metric.num_false_negatives = int(
            sum((all_train_labels == 1) & (all_train_preds == 0))
        )

        train_metric.avg_loss = float(
            train_metric.total_loss / train_metric.dataset_size
        )

        if train_metric.dataset_size != 0:
            train_metric.accuracy = float(
                (train_metric.num_true_positives + train_metric.num_true_negatives)
                / train_metric.dataset_size
            )

        if train_metric.num_true_positives + train_metric.num_false_positives != 0:
            train_metric.precision = float(
                train_metric.num_true_positives
                / (train_metric.num_true_positives + train_metric.num_false_positives)
            )

        if train_metric.num_true_positives + train_metric.num_false_negatives != 0:
            train_metric.recall = float(
                train_metric.num_true_positives
                / (train_metric.num_true_positives + train_metric.num_false_negatives)
            )

        if (train_metric.precision + train_metric.recall) != 0:
            train_metric.f1_score = float(
                2
                * (train_metric.precision * train_metric.recall)
                / (train_metric.precision + train_metric.recall)
            )

        if (epoch + 1) % config.log_interval == 0:
            logger.info(f"Training loss: {train_metric.avg_loss}")
            logger.info(f"Training accuracy: {train_metric.accuracy}")

        val_metric = None

        if (epoch + 1) % config.eval_interval == 0:
            val_metric = evaluate_model(
                model,
                config.val_dataloader,
                config.loss_function,
                config.use_cuda,
            )
            logger.info(f"Validation loss: {val_metric.avg_loss}")
            logger.info(f"Validation accuracy: {val_metric.accuracy}")

        history.add_epoch(
            epoch_num=epoch,
            train_metric=train_metric,
            val_metric=val_metric,
        )

        if (epoch + 1) % 5 == 0:
            checkpoint_name = (
                f"{model.name}_{config.get_config_unique_name()}_epoch{epoch+1}.pth"
            )
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/{checkpoint_name}")
            logger.info(f"Checkpointing model to {checkpoint_name}")

    model_name = f"{model.name}_{config.get_config_unique_name()}.pth"
    os.makedirs("history/models", exist_ok=True)
    torch.save(model.state_dict(), f"history/models/{model_name}")
    logger.info(f"Saving model to {model_name}")

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="Number of epochs to train for",
    )
    args = parser.parse_args()

    model = FakeNewsDetector()

    train_dataloader, val_dataloader = get_dataloaders(
        _DATASET_PTH,
        train_size=0.8,
        batch_size=128,
        max_records=1000,
    )

    config = TrainingConfig(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=args.epochs,
        batch_size=128,
        learning_rate=0.001,
        weight_decay=0.0001,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        loss_function=nn.BCELoss(),
        use_cuda=torch.cuda.is_available(),
        log_interval=1,
        eval_interval=args.epochs // 10,
    )

    history = train_model(model, config)
    os.makedirs("history/training_history", exist_ok=True)
    history_dict = asdict(history)
    history_file_name = (
        f"{model.name}_{int(time.time() * 1000)}_{config.get_config_unique_name()}.json"
    )
    with open(
        f"history/training_history/{history_file_name}",
        "w",
    ) as f:
        logger.info(f"Saving history to {history_file_name}")
        json.dump(history_dict, f)


if __name__ == "__main__":
    main()
