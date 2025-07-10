import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
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

    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(y_test, y_pred, model, X_test_tfidf)


def create_visualizations(y_true, y_pred, model, X_test_tfidf):
    """Create comprehensive visualizations for model performance evaluation."""

    # Set up the plotting style
    plt.style.use("default")
    sns.set_palette("husl")

    # Create a figure with subplots (2 rows, 2 columns instead of 2x3)
    fig = plt.figure(figsize=(16, 12))

    # 1. Confusion Matrix
    plt.subplot(2, 3, 1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["True News", "Fake News"],
        yticklabels=["True News", "Fake News"],
    )
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    # 2. Performance Metrics Bar Chart
    plt.subplot(2, 3, 2)
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    values = [accuracy_score(y_true, y_pred), precision, recall, f1_score]
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]

    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.title("Model Performance Metrics", fontsize=14, fontweight="bold")
    plt.ylabel("Score")
    plt.ylim(0, 1)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

        # 3. Precision-Recall Curve
    plt.subplot(2, 2, 3)
    y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1]
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall_curve, precision_curve)

    plt.plot(
        recall_curve,
        precision_curve,
        color="blue",
        lw=2,
        label=f"PR curve (AUC = {pr_auc:.3f})",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)

    # 4. Class Balance Check
    plt.subplot(2, 2, 4)
    class_counts = np.bincount(y_true)
    total_samples = len(y_true)
    labels = [
        f"True News\n({class_counts[0]:,} samples)",
        f"Fake News\n({class_counts[1]:,} samples)",
    ]
    colors = ["#2E86AB", "#A23B72"]

    plt.pie(
        class_counts, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90
    )
    plt.title(
        f"Test Set Class Distribution\n(Total: {total_samples:,} samples)",
        fontsize=14,
        fontweight="bold",
    )

    # Create visualizations directory if it doesn't exist
    viz_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "visualizations"
    )
    os.makedirs(viz_dir, exist_ok=True)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(
        os.path.join(viz_dir, "model_performance_visualizations.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()  # Close the figure to free memory

    # Print detailed metrics
    print(f"\nDetailed Performance Metrics:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")
    print(
        f"\nVisualizations saved as '{os.path.join(viz_dir, 'model_performance_visualizations.png')}'"
    )
