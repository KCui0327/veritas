from model import model_inference, train_model


def main():
    train_model()
    model_inference("This is a fake news article")


if __name__ == "__main__":
    main()
