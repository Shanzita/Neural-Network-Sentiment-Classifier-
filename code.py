#!/usr/bin/env python3
"""
sentiment_nn.py
---------------
End-to-end sentiment classifier for Amazon book reviews using TF-IDF + Keras.

Usage:
  python sentiment_nn.py --data ./data_NLP/bookReviews.csv --epochs 55 --lr 0.1 --optimizer sgd
  python sentiment_nn.py --data ./data_NLP/bookReviews.csv --epochs 25 --optimizer adam --dropout 0.3

Expected CSV columns:
  - "Review": free-text review
  - "Positive Review": boolean or {0,1} label (1=positive, 0=negative)

Outputs:
  - Prints dataset stats, model summary, evaluation metrics
  - Saves training curves to: training_loss.png, training_acc.png
  - Saves model to: ./model/
  - Saves vectorizer to: ./model/tfidf.pkl
"""

from __future__ import annotations
import argparse
import os
import random
import pickle
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def set_seeds(seed: int = 1234) -> None:
    """Fix seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load the reviews CSV and perform light cleaning."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # Standardize column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]
    if "Review" not in df.columns or "Positive Review" not in df.columns:
        raise KeyError('CSV must have "Review" and "Positive Review" columns.')

    # Drop rows with missing text/label
    df = df.dropna(subset=["Review", "Positive Review"]).copy()

    # Normalize labels to {0,1}
    def to_binary(val):
        if isinstance(val, (int, float, np.integer, np.floating)):
            return int(val > 0.5)
        if isinstance(val, str):
            v = val.strip().lower()
            if v in {"true", "yes", "y", "1", "positive"}:
                return 1
            if v in {"false", "no", "n", "0", "negative"}:
                return 0
            try:
                return int(float(v) > 0.5)
            except Exception:
                pass
        if isinstance(val, (bool, np.bool_)):
            return int(bool(val))
        raise ValueError(f"Unrecognized label value: {val!r}")

    df["Positive Review"] = df["Positive Review"].apply(to_binary).astype(int)
    df["Review"] = df["Review"].astype(str)
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.25,
    seed: int = 1234,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Split into train/test."""
    X = df["Review"]
    y = df["Positive Review"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test


def build_vectorizer(
    max_features: Optional[int] = None,
    min_df: int = 1,
    ngram_max: int = 1,
) -> TfidfVectorizer:
    """Configure a TF-IDF vectorizer."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, ngram_max),
        min_df=min_df,
        lowercase=True,
        strip_accents="unicode",
    )


def vectorize_fit_transform(
    vec: TfidfVectorizer, X_train: pd.Series, X_test: pd.Series
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit on train, transform both; convert to dense arrays for Keras."""
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)
    return X_train_vec.toarray(), X_test_vec.toarray()

def build_model(
    input_dim: int,
    hidden: Tuple[int, int, int] = (64, 32, 16),
    dropout: float = 0.0,
    optimizer: str = "sgd",
    lr: float = 0.1,
) -> keras.Model:
    """Create a simple feed-forward network for binary classification."""
    model = keras.Sequential(name="tfidf_ffnn")
    model.add(layers.Input(shape=(input_dim,), name="tfidf_input"))
    for i, units in enumerate(hidden):
        model.add(layers.Dense(units, activation="relu", name=f"dense_{i+1}"))
        if dropout and dropout > 0:
            model.add(layers.Dropout(dropout, name=f"dropout_{i+1}"))
    model.add(layers.Dense(1, activation="sigmoid", name="output"))

    if optimizer.lower() == "adam":
        opt = keras.optimizers.Adam(learning_rate=lr)
    else:
        opt = keras.optimizers.SGD(learning_rate=lr)

    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: pd.Series,
    epochs: int = 55,
    batch_size: int = 32,
    val_split: float = 0.2,
    patience: Optional[int] = None,
) -> keras.callbacks.History:
    """Train with optional early stopping."""
    callbacks = []
    if patience is not None and patience > 0:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience, restore_best_weights=True
            )
        )
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        verbose=1,
        callbacks=callbacks,
    )
    return history


def evaluate_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict:
    """Evaluate on test set and return metrics dict."""
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    probs = model.predict(X_test, verbose=0).ravel()
    y_pred = (probs >= threshold).astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Test Metrics ===")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    return {
        "loss": float(loss),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
    }
def plot_curves(history: keras.callbacks.History, out_dir: str = ".") -> None:
    """Save training/validation loss and accuracy plots (separate figures)."""
    hist = history.history

    # Loss plot
    plt.figure()
    plt.plot(hist.get("loss", []), label="train_loss")
    plt.plot(hist.get("val_loss", []), label="val_loss")
    plt.title("Loss by Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    loss_path = os.path.join(out_dir, "training_loss.png")
    plt.savefig(loss_path, bbox_inches="tight")
    plt.close()

    # Accuracy plot
    plt.figure()
    plt.plot(hist.get("accuracy", []), label="train_acc")
    plt.plot(hist.get("val_accuracy", []), label="val_acc")
    plt.title("Accuracy by Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    acc_path = os.path.join(out_dir, "training_acc.png")
    plt.savefig(acc_path, bbox_inches="tight")
    plt.close()

    print(f"\nSaved plots to:\n  {loss_path}\n  {acc_path}")


def save_artifacts(model: keras.Model, vectorizer: TfidfVectorizer, out_dir: str = "./model") -> None:
    """Save Keras model and TF-IDF vectorizer."""
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "keras_model")
    vec_path = os.path.join(out_dir, "tfidf.pkl")
    model.save(model_path)  # SavedModel format
    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"\nSaved model to: {model_path}\nSaved vectorizer to: {vec_path}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TF-IDF + Keras Sentiment Classifier")
    p.add_argument("--data", type=str, required=True, help="Path to reviews CSV (bookReviews.csv)")
    p.add_argument("--epochs", type=int, default=55, help="Training epochs")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size")
    p.add_argument("--val_split", type=float, default=0.2, help="Validation split (0..1)")
    p.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    p.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"], help="Optimizer")
    p.add_argument("--dropout", type=float, default=0.0, help="Dropout after each hidden layer (0..1)")
    p.add_argument("--hidden", type=str, default="64,32,16", help="Comma-separated hidden sizes, e.g., 128,64")
    p.add_argument("--max_features", type=int, default=None, help="TF-IDF max features (None=all)")
    p.add_argument("--min_df", type=int, default=1, help="Ignore terms with doc freq < min_df")
    p.add_argument("--ngram_max", type=int, default=1, help="Use up to this n-gram size (1=unigram, 2=bi-gram)")
    p.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for positive class")
    p.add_argument("--seed", type=int, default=1234, help="Random seed")
    p.add_argument("--save_dir", type=str, default="./model", help="Where to save model/vectorizer")
    return p.parse_args()


def main():
    args = parse_args()
    set_seeds(args.seed)

    # Parse hidden sizes
    hidden = tuple(int(x) for x in args.hidden.split(",") if x.strip())

    print("=== Configuration ===")
    print(vars(args))

    # Load & split
    df = load_dataset(args.data)
    print(f"\nLoaded {len(df):,} rows from {args.data}")
    print(df.head(3))
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.25, seed=args.seed)
    print(f"\nTrain size: {len(X_train):,} | Test size: {len(X_test):,}")

    # Vectorize
    vectorizer = build_vectorizer(
        max_features=args.max_features,
        min_df=args.min_df,
        ngram_max=args.ngram_max,
    )
    X_train_vec, X_test_vec = vectorize_fit_transform(vectorizer, X_train, X_test)
    input_dim = X_train_vec.shape[1]
    print(f"\nTF-IDF vocabulary size: {input_dim:,}")

    # Build model
    model = build_model(
        input_dim=input_dim,
        hidden=hidden,
        dropout=args.dropout,
        optimizer=args.optimizer,
        lr=args.lr,
    )
    model.summary()

    # Train
    history = train_model(
        model,
        X_train_vec, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_split=args.val_split,
        patience=None,  # set to e.g., 5 to enable early stopping
    )

    # Visualize
    plot_curves(history, out_dir=".")

    # Evaluate
    metrics = evaluate_model(model, X_test_vec, y_test, threshold=args.threshold)

    # Save artifacts
    save_artifacts(model, vectorizer, out_dir=args.save_dir)

    # Final summary
    print("\n=== Done ===")
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()})


if __name__ == "__main__":
    main()
