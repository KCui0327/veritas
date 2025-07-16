import argparse
import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from src.data_models.evaluation_metric import EvaluationMetric
from src.dataset.get_data_loaders import get_dataloaders


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

    model.eval()
    if use_cuda:
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

        metric.avg_loss = metric.total_loss / len(dataloader)
        metric.inference_time = time.time() - start_time

    metric.dataset_num_true_labels = sum(all_train_labels)
    metric.dataset_num_false_labels = (
        len(all_train_labels) - metric.dataset_num_true_labels
    )

    all_train_labels = np.array(all_train_labels)
    all_train_preds = np.array(all_train_preds)

    metric.num_true_positives = sum((all_train_labels == 1) & (all_train_preds == 1))
    metric.num_false_positives = sum((all_train_labels == 0) & (all_train_preds == 1))
    metric.num_true_negatives = sum((all_train_labels == 0) & (all_train_preds == 0))
    metric.num_false_negatives = sum((all_train_labels == 1) & (all_train_preds == 0))

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
    _, validation_loader = get_dataloaders()

    use_cuda = torch.cuda.is_available()

    evaluate_model(model, validation_loader, criterion, use_cuda)
