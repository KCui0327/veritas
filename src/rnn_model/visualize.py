from src.rnn_model.training_history import TrainingHistory

_CHECKPOINT_PATH = "checkpoints/FakeNewsDetector_32_0.001_0.0.json"
import os

_VISUALIZATION_PATH = "visualizations/"


def main():
    history = TrainingHistory()
    history.load(_CHECKPOINT_PATH)
    history.plot_training_curves(save_path="visualizations/training_curves.png")


if __name__ == "__main__":
    main()
