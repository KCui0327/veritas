from src.rnn_model.training_history import TrainingHistory
import os
import sys

_VISUALIZATION_PATH = "visualizations/"


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
