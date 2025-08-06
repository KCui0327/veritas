from dataclasses import dataclass, field
from typing import Dict

from src.data_models.evaluation_metric import EvaluationMetric


@dataclass
class TrainingHistory:
    """Class to track training history for visualization."""

    train_metrics: Dict[int, EvaluationMetric] = field(default_factory=dict)
    val_metrics: Dict[int, EvaluationMetric] = field(default_factory=dict)

    def add_epoch(
        self,
        epoch_num: int,
        train_metric: EvaluationMetric,
        val_metric: EvaluationMetric = None,
    ):
        self.train_metrics[epoch_num] = train_metric
        if val_metric is not None:
            self.val_metrics[epoch_num] = val_metric
