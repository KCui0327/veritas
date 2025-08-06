from dataclasses import dataclass


@dataclass
class EvaluationMetric:
    num_true_positives: int = 0
    num_false_positives: int = 0
    num_true_negatives: int = 0
    num_false_negatives: int = 0

    accuracy: float = 0
    precision: float = 0
    recall: float = 0
    f1_score: float = 0

    inference_time: float = 0
    num_model_parameters: int = 0
    total_loss: float = 0
    avg_loss: float = 0
    learning_rate: float = 0

    dataset_size: int = 0
    batch_size: int = 0
    dataset_num_true_labels: int = 0
    dataset_num_false_labels: int = 0

    time_taken: float = 0
