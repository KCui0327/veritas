import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

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

    y_pred = model.predict(X_test_tfidf)

    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test, y_pred, target_names=["True News", "Fake News"], zero_division=0
        )
    )

    print("\nSaving model...")
    with open(MODEL_FILE, "wb") as f:
        pickle.dump((vectorizer, model), f)

    print("=== Training Complete ===")
    print(f"Model saved as '{MODEL_FILE}'")
