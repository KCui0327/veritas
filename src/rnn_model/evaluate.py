import argparse
import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from src.data_models.evaluation_metric import EvaluationMetric
from src.dataset.get_data_loaders import get_dataloaders
from src.util.logger import logger


_DATASET_PTH = "data/veritas_dataset.csv"


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: Callable,
    use_cuda: bool,
) -> EvaluationMetric:
    """
    Evaluate the model with the given dataloader and criterion.

    This can be called directly by loading a trained model, or as part of the
    training process.
    """

    logger.info("Evaluating model")
    model.eval()
    if use_cuda:
        logger.info("Using CUDA")
        model.cuda()

    metric = EvaluationMetric(
        dataset_size=len(dataloader.dataset),
        batch_size=dataloader.batch_size,
        num_model_parameters=sum(p.numel() for p in model.parameters()),
    )

    all_train_labels = []
    all_train_preds = []

    with torch.no_grad():
        start_time = time.time()

        for batch in dataloader:
            inputs, labels = batch
            if use_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = model(inputs)
            preds = (outputs >= 0.5).float()

            loss = criterion(outputs, labels)
            metric.total_loss += loss.item()

            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(preds.cpu().numpy())

    all_train_labels = np.array(all_train_labels)
    all_train_preds = np.array(all_train_preds)

    metric.avg_loss = float(metric.total_loss / metric.dataset_size)
    metric.inference_time = time.time() - start_time
    metric.dataset_num_true_labels = int(sum(all_train_labels))
    metric.dataset_num_false_labels = int(
        len(all_train_labels) - metric.dataset_num_true_labels
    )

    metric.num_true_positives = int(
        sum((all_train_labels == 1) & (all_train_preds == 1))
    )
    metric.num_false_positives = int(
        sum((all_train_labels == 0) & (all_train_preds == 1))
    )
    metric.num_true_negatives = int(
        sum((all_train_labels == 0) & (all_train_preds == 0))
    )
    metric.num_false_negatives = int(
        sum((all_train_labels == 1) & (all_train_preds == 0))
    )

    if metric.dataset_size != 0:
        metric.accuracy = float(
            (metric.num_true_positives + metric.num_true_negatives)
            / metric.dataset_size
        )

    if metric.num_true_positives + metric.num_false_positives != 0:
        metric.precision = float(
            metric.num_true_positives
            / (metric.num_true_positives + metric.num_false_positives)
        )

    if metric.num_true_positives + metric.num_false_negatives != 0:
        metric.recall = float(
            metric.num_true_positives
            / (metric.num_true_positives + metric.num_false_negatives)
        )

    if (metric.precision + metric.recall) != 0:
        metric.f1_score = float(
            2 * (metric.precision * metric.recall) / (metric.precision + metric.recall)
        )

    return metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model file",
    )
    args = parser.parse_args()

    model_path = args.model_path
    model = torch.load(model_path)
    criterion = nn.CrossEntropyLoss()
    _, validation_loader = get_dataloaders(_DATASET_PTH)

    use_cuda = torch.cuda.is_available()

    evaluate_model(model, validation_loader, criterion, use_cuda)
