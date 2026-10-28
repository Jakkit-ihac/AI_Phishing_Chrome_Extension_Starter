import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
SPLIT_DIR = PROJECT_ROOT / "dataset" / "splits"
MODEL_DIR = BACKEND_DIR / "models"
REPORT_DIR = BACKEND_DIR / "reports"

sys.path.insert(0, str(BACKEND_DIR))

from app.ml.features import extract_features


RANDOM_STATE = 42
THRESHOLD = 0.5


def load_split(name):
    path = SPLIT_DIR / f"{name}.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")

    df = pd.read_csv(path)

    if not {"url", "label"}.issubset(df.columns):
        raise ValueError(f"Invalid columns in {path}")

    df = df.dropna(subset=["url", "label"]).copy()
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(f"Unexpected labels in {path}")

    if set(df["label"].unique()) != {0, 1}:
        raise ValueError(f"Both labels 0 and 1 are required in {path}")

    return df.reset_index(drop=True)


def build_features(df, split_name):
    rows = []
    valid_indices = []
    errors = []

    for index, url in enumerate(df["url"]):
        try:
            rows.append(extract_features(str(url)))
            valid_indices.append(index)
        except (ValueError, TypeError) as exc:
            errors.append((index, str(exc)))

    if errors:
        print(
            f"WARNING: {len(errors)} invalid URLs skipped "
            f"from {split_name}."
        )
        for index, reason in errors[:3]:
            print(f"  Row {index}: {reason}")

    if not rows:
        raise ValueError(f"No valid URLs in {split_name}")

    X = pd.DataFrame(rows)
    y = df.iloc[valid_indices]["label"].reset_index(drop=True)

    return X, y


def get_metrics(model, X, y):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    return {
        "threshold": THRESHOLD,
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(
            y, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y, predictions, zero_division=0
        )),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "pr_auc_average_precision": float(
            average_precision_score(y, probabilities)
        ),
        "confusion_matrix": confusion_matrix(
            y, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=["legitimate_0", "phishing_1"],
            output_dict=True,
            zero_division=0,
        ),
    }


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading train and validation datasets...")
    train_df = load_split("train")
    validation_df = load_split("validation")

    print(f"Train rows: {len(train_df):,}")
    print(f"Validation rows: {len(validation_df):,}")

    print("\nExtracting URL features...")
    X_train, y_train = build_features(train_df, "train")
    X_val, y_val = build_features(validation_df, "validation")

    if list(X_train.columns) != list(X_val.columns):
        raise RuntimeError("Feature columns do not match")

    feature_names = list(X_train.columns)

    print(f"Feature count: {len(feature_names)}")
    print(f"Training samples used: {len(X_train):,}")
    print(f"Validation samples used: {len(X_val):,}")

    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=RANDOM_STATE,
            )),
        ]),
        "random_forest": Pipeline([
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                min_samples_leaf=2,
            )),
        ]),
    }

    results = {}
    best_name = None
    best_model = None
    best_score = float("-inf")

    for name, model in models.items():
        print("\n" + "=" * 60)
        print(f"Training: {name}")
        print("=" * 60)

        start = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - start

        metrics = get_metrics(model, X_val, y_val)
        metrics["training_seconds"] = round(elapsed, 2)
        metrics["feature_count"] = len(feature_names)
        results[name] = metrics

        print(f"Training time: {elapsed:.2f} seconds")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-score:  {metrics['f1']:.4f}")
        print(
            "PR-AUC (Average Precision): "
            f"{metrics['pr_auc_average_precision']:.4f}"
        )
        print("Confusion matrix [[TN, FP], [FN, TP]]:")
        print(metrics["confusion_matrix"])

        # Select based on validation PR-AUC, not training accuracy.
        score = metrics["pr_auc_average_precision"]

        if score > best_score:
            best_score = score
            best_name = name
            best_model = model

    print("\n" + "=" * 60)
    print(f"Best model on validation: {best_name}")
    print("=" * 60)

    model_path = MODEL_DIR / "best_model.joblib"
    joblib.dump(best_model, model_path)

    metadata = {
        "model_name": best_name,
        "positive_label": 1,
        "positive_label_meaning": "phishing",
        "negative_label": 0,
        "negative_label_meaning": "legitimate",
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "decision_threshold": THRESHOLD,
        "selection_metric": "validation PR-AUC (average precision)",
        "validation_pr_auc": best_score,
        "probability_calibration": "not calibrated",
        "dataset_file": "dataset/cleaned_urls.csv",
        "split_seed": RANDOM_STATE,
        "models_compared": list(models.keys()),
    }

    metadata_path = MODEL_DIR / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report_path = REPORT_DIR / "validation_metrics.json"
    report_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    comparison = []

    for name, metrics in results.items():
        comparison.append({
            "model": name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "pr_auc_average_precision":
                metrics["pr_auc_average_precision"],
            "training_seconds": metrics["training_seconds"],
        })

    comparison_path = REPORT_DIR / "model_comparison.csv"
    pd.DataFrame(comparison).to_csv(comparison_path, index=False)

    print("\nSaved artifacts:")
    print(f"Model: {model_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Validation report: {report_path}")
    print(f"Comparison CSV: {comparison_path}")
    print("\nTraining complete. Test set has not been used.")


if __name__ == "__main__":
    main()
