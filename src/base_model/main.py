from model import model
from train import train_model

from src.dataset import split_dataset


def main():
    X_train, X_test, y_train, y_test = split_dataset()
    train_model(model, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
