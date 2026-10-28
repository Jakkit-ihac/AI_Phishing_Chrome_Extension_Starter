import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
MODEL_PATH = BACKEND_DIR / "models" / "best_model.joblib"
METADATA_PATH = BACKEND_DIR / "models" / "metadata.json"
TEST_PATH = PROJECT_ROOT / "dataset" / "splits" / "test.csv"
REPORT_DIR = BACKEND_DIR / "reports"

sys.path.insert(0, str(BACKEND_DIR))

from app.ml.features import extract_features


def main():
    for path in [MODEL_PATH, METADATA_PATH, TEST_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    # Only load model artifacts created by your own training script.
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    df = pd.read_csv(TEST_PATH).dropna(subset=["url", "label"]).copy()
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

    rows = []
    labels = []
    skipped = 0

    for url, label in zip(df["url"], df["label"]):
        try:
            features = extract_features(str(url))
            rows.append(features)
            labels.append(int(label))
        except (ValueError, TypeError) as exc:
            skipped += 1
            if skipped <= 5:
                print(f"Skipping invalid URL: {exc}")

    if not rows:
        raise ValueError("No valid URLs available for evaluation.")

    X_test = pd.DataFrame(rows)
    y_test = pd.Series(labels, dtype="int64")

    expected_features = metadata["feature_names"]

    if list(X_test.columns) != expected_features:
        raise RuntimeError(
            "Feature schema mismatch between model metadata and extractor."
        )

    threshold = float(metadata["decision_threshold"])
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    results = {
        "model_name": metadata["model_name"],
        "dataset": "held-out test set",
        "threshold": threshold,
        "samples_evaluated": int(len(y_test)),
        "invalid_urls_skipped": int(skipped),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "pr_auc_average_precision": float(
            average_precision_score(y_test, probabilities)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["legitimate_0", "phishing_1"],
            output_dict=True,
            zero_division=0,
        ),
        "probability_calibration": "not calibrated",
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "test_metrics.json"
    report_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("HELD-OUT TEST SET EVALUATION")
    print("=" * 60)
    print(f"Model: {results['model_name']}")
    print(f"Samples evaluated: {results['samples_evaluated']:,}")
    print(f"Invalid URLs skipped: {results['invalid_urls_skipped']}")
    print(f"Threshold: {threshold}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-score:  {results['f1']:.4f}")
    print(
        "PR-AUC (Average Precision): "
        f"{results['pr_auc_average_precision']:.4f}"
    )
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(results["confusion_matrix"])
    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["legitimate_0", "phishing_1"],
        zero_division=0,
    ))
    print(f"\nSaved test report: {report_path}")
    print("Evaluation complete. Model parameters were not changed.")


if __name__ == "__main__":
    main()
