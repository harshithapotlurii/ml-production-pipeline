from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ("tenure_months", "monthly_spend", "support_tickets")


def synthetic_data(n: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, size=n)
    spend = rng.uniform(15, 180, size=n)
    tickets = rng.poisson(2, size=n)
    x = np.column_stack((tenure, spend, tickets)).astype(float)
    logits = -0.9 - 0.035 * tenure + 0.012 * spend + 0.45 * tickets
    probabilities = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, probabilities)
    return x, y


def train(output: Path, *, seed: int = 42) -> dict:
    x, y = synthetic_data(seed=seed)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=seed, stratify=y
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=500, random_state=seed)),
    ])
    model.fit(x_train, y_train)
    scores = model.predict_proba(x_test)[:, 1]
    labels = (scores >= 0.5).astype(int)
    metrics = {
        "dataset": "synthetic",
        "seed": seed,
        "train_rows": len(y_train),
        "test_rows": len(y_test),
        "test_positive_rate": float(np.mean(y_test)),
        "accuracy": float(accuracy_score(y_test, labels)),
        "precision": float(precision_score(y_test, labels, zero_division=0)),
        "recall": float(recall_score(y_test, labels, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, scores)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, output)
    output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("model.joblib"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(train(args.output, seed=args.seed), indent=2))
