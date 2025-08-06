import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(MODEL_DIR, "base_model.pkl")


def train_model(
    model: LogisticRegression,
    X_train: list[str],
    y_train: list[int],
    X_test: list[str],
    y_test: list[int],
):
    """Main training function using scikit-learn's LogisticRegression directly."""
    print("=== Fake News Detection with Logistic Regression ===\n")

    print("Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Training Logistic Regression model...")
    model.fit(X_train_tfidf, y_train)

    y_pred_train = model.predict(X_train_tfidf)
    y_pred_test = model.predict(X_test_tfidf)

    print(f"Test Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred_test,
            target_names=["True News", "Fake News"],
            zero_division=0,
        )
    )

    print("\nSaving model...")
    with open(MODEL_FILE, "wb") as f:
        pickle.dump((y_train, y_pred_train, y_test, y_pred_test, model), f)

        print("=== Training Complete ===")
    print(f"Model saved as '{MODEL_FILE}'")


def main():
    data_frame = pd.read_csv("src/dataset/veritas_dataset.csv")

    data_frame = data_frame.sample(frac=1).reset_index(drop=True)

    train_size = int(0.8 * len(data_frame))
    val_size = len(data_frame) - train_size

    train_data_frame = data_frame.iloc[:train_size]
    val_data_frame = data_frame.iloc[train_size:]

    # map the verdict to 1 or 0
    train_data_frame["verdict"] = train_data_frame["verdict"].map({True: 0, False: 1})
    val_data_frame["verdict"] = val_data_frame["verdict"].map({True: 0, False: 1})

    train_model(
        model,
        train_data_frame["statement"],
        train_data_frame["verdict"],
        val_data_frame["statement"],
        val_data_frame["verdict"],
    )


if __name__ == "__main__":
    main()
