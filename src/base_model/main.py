from model import model
from train import model_inference, train_model


def main():
    # TODO: use dataloader
    X_train, y_train, X_test, y_test = None, None, None, None
    train_model(model, X_train, y_train, X_test, y_test)

    model_inference(model, "This is a fake news article")

    # TODO: Confusion matrix and visualizations


if __name__ == "__main__":
    main()
