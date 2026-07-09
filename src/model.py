"""
Lightweight, self-contained classifier for the CLI / Streamlit demo.

This is the same ML approach as the full MLflow pipeline in src/train.py
(TF-IDF features + LogisticRegression), but it trains and loads from simple
local .pkl files so the demo can run end-to-end without an MLflow server or the
full Docker stack. It reuses the shared preprocessing/vectorization helpers in
src/preprocessing.py, so the modelling stays consistent across both paths.

Typical use:

    from src.model import train_and_save, classify_and_triage
    train_and_save()                      # writes models/*.pkl (once)
    result = classify_and_triage("My VPN won't connect from home")
    print(result["routing_team"])         # -> "Network Operations"
"""

import os
import sys
from typing import List, Optional, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    MODELS_DIR,
    MODEL_FILE,
    VECTORIZER_FILE,
    LABELS,
)
from src.data_loader import load_support_ticket_data, train_val_test_split
from src.preprocessing import preprocess_texts, fit_vectorizer
from src.evaluation import evaluate_classification_model
from src.service_desk import triage_ticket

# Cache so we only load the model/vectorizer from disk once per process.
_model = None
_vectorizer = None


def train_and_save(verbose: bool = True) -> dict:
    """
    Train the TF-IDF + LogisticRegression classifier and save it locally.

    Loads the IT ticket dataset, splits it, fits the vectorizer and model,
    evaluates on the held-out test set, and writes the model and vectorizer to
    the models/ directory.

    Returns:
        A dict of test-set metrics (accuracy, macro/weighted F1, ...).
    """
    df = load_support_ticket_data()
    train_df, val_df, test_df = train_val_test_split(df)

    train_texts = preprocess_texts(train_df["text"].tolist())
    test_texts = preprocess_texts(test_df["text"].tolist())

    # Small dataset, so use min_df=1 to keep useful rare keywords.
    vectorizer = fit_vectorizer(
        train_texts,
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
    )
    X_train = vectorizer.transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    # C=10 (lighter regularization) sharpens the softmax so confident, keyword-
    # clear tickets score ~0.8 while genuinely ambiguous ones stay low - which is
    # what the service desk layer uses to flag tickets for manual review.
    model = LogisticRegression(C=10.0, max_iter=2000, random_state=42)
    model.fit(X_train, train_df["label"].tolist())

    y_test_pred = model.predict(X_test)
    metrics = evaluate_classification_model(
        test_df["label"].tolist(), y_test_pred, labels=LABELS
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)

    # Refresh the in-process cache with what we just trained.
    global _model, _vectorizer
    _model, _vectorizer = model, vectorizer

    if verbose:
        print(f"Saved model to      {MODEL_FILE}")
        print(f"Saved vectorizer to {VECTORIZER_FILE}")
        print(f"Test accuracy : {metrics['accuracy']:.3f}")
        print(f"Test macro-F1 : {metrics['macro_f1']:.3f}")

    return metrics


def load_model(auto_train: bool = True) -> Tuple[LogisticRegression, object]:
    """
    Load the saved model and vectorizer, training them first if missing.

    Args:
        auto_train: If True and no saved model exists, train one automatically.

    Returns:
        (model, vectorizer)
    """
    global _model, _vectorizer
    if _model is not None and _vectorizer is not None:
        return _model, _vectorizer

    if not (MODEL_FILE.exists() and VECTORIZER_FILE.exists()):
        if not auto_train:
            raise FileNotFoundError(
                f"No saved model at {MODEL_FILE}. Run: python -m src.model"
            )
        print("No saved model found - training a new one...")
        train_and_save()
        return _model, _vectorizer

    _model = joblib.load(MODEL_FILE)
    _vectorizer = joblib.load(VECTORIZER_FILE)
    return _model, _vectorizer


def classify(texts: List[str]) -> List[Tuple[str, float]]:
    """
    Predict the IT category and confidence for each ticket text.

    Returns:
        A list of (predicted_category, confidence) tuples.
    """
    model, vectorizer = load_model()
    features = vectorizer.transform(preprocess_texts(texts))

    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    classes = list(model.classes_)

    results = []
    for pred, proba in zip(predictions, probabilities):
        confidence = float(proba[classes.index(pred)])
        results.append((pred, confidence))
    return results


def classify_and_triage(text: str) -> dict:
    """
    Full pipeline for a single ticket: ML prediction + service desk triage.

    This is the function the CLI and Streamlit demos call. It returns the
    complete triage dictionary (category, confidence, priority, routing team,
    suggested KB, first steps, escalation).
    """
    category, confidence = classify([text])[0]
    return triage_ticket(text, category, confidence)


if __name__ == "__main__":
    # `python -m src.model` (re)trains and saves the demo model.
    train_and_save()
