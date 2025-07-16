import os
import sys

from src.data_models.evaluation_metric import EvaluationMetric
from src.data_models.training_history import TrainingHistory

_VISUALIZATION_PATH = "visualizations/"


def visualize_training_history(history: TrainingHistory, save_path: str):
    pass


def visualize_evaluation_metric(metric: EvaluationMetric, save_path: str):
    pass


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.rnn_model.visualize <training_history_path>")
        exit(1)

    history = TrainingHistory()
    history.load(sys.argv[1])
    history.plot_training_curves(
        save_path=f"visualizations/{os.path.basename(sys.argv[1])}.png"
    )


if __name__ == "__main__":
    main()
