import os
import pickle
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(MODEL_DIR, "base_model.pkl")


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def create_sample_data():
    """Create sample training data."""
    true_news = [
        "Scientists discover new species of deep-sea creatures in the Pacific Ocean.",
        "NASA successfully launches new satellite to monitor climate change.",
        "Major tech company announces breakthrough in renewable energy storage.",
        "International team develops new treatment for rare genetic disorder.",
        "Global economic summit addresses climate change policies.",
    ]

    fake_news = [
        "Aliens spotted in downtown area, government covering up evidence.",
        "Secret cure for all diseases discovered but hidden by big pharma.",
        "Time travel machine invented by backyard scientist.",
        "Moon landing was completely faked, new evidence proves.",
        "Giant sea monsters discovered in Atlantic Ocean.",
    ]

    texts = true_news + fake_news
    labels = [0] * len(true_news) + [1] * len(fake_news)  # 0 for true, 1 for fake
    return texts, labels


def train_model():
    """Main training function using scikit-learn's LogisticRegression directly."""
    print("=== Fake News Detection with Logistic Regression ===\n")

    print("Loading training data...")
    texts, labels = create_sample_data()
    processed_texts = [preprocess_text(text) for text in texts]

    X_train, X_test, y_train, y_test = train_test_split(
        processed_texts, labels, test_size=0.3, random_state=42, stratify=labels
    )

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
    model = LogisticRegression(random_state=42, max_iter=1000, solver="liblinear")
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test, y_pred, target_names=["True News", "Fake News"], zero_division=0
        )
    )

    print("\n=== Testing on New Articles ===")
    test_articles = [
        "New study shows benefits of regular exercise on mental health.",
        "Secret government program creates mind-control devices.",
        "Breakthrough in quantum computing announced by research team.",
        "Bigfoot DNA evidence finally confirmed by scientists.",
    ]

    for i, article in enumerate(test_articles, 1):
        # Preprocess and vectorize
        processed_article = preprocess_text(article)
        article_tfidf = vectorizer.transform([processed_article])

        # Get probabilities (softmax)
        proba = model.predict_proba(article_tfidf)[0]
        prediction = model.predict(article_tfidf)[0]

        print(f"\nArticle {i}: {article[:50]}...")
        print(f"True News Probability: {proba[0]:.4f}")
        print(f"Fake News Probability: {proba[1]:.4f}")
        print(f"Prediction: {'Fake News' if prediction == 1 else 'True News'}")

    # Save model and vectorizer
    print("\nSaving model...")
    with open(MODEL_FILE, "wb") as f:
        pickle.dump((vectorizer, model), f)

    print("=== Training Complete ===")
    print(f"Model saved as '{MODEL_FILE}'")


def model_inference(text: str, model_file: str = MODEL_FILE):
    """
    Predict fake news probability for a given text.

    Args:
        text: Input news article text
        model_file: Path to saved model file

    Returns:
        Tuple of (true_news_prob, fake_news_prob, prediction)
    """
    # Load model
    with open(model_file, "rb") as f:
        vectorizer, model = pickle.load(f)

    # Preprocess and predict
    processed_text = preprocess_text(text)
    text_tfidf = vectorizer.transform([processed_text])

    proba = model.predict_proba(text_tfidf)[0]
    prediction = model.predict(text_tfidf)[0]

    return proba[0], proba[1], prediction
