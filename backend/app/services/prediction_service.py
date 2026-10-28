import json
import logging
import math
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import pandas as pd

from app.ml.features import extract_features
from app.schemas.prediction import (
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BACKEND_DIR / "models" / "best_model.joblib"
METADATA_PATH = BACKEND_DIR / "models" / "metadata.json"

_model = None
_metadata = None


def load_model() -> bool:
    """Load the saved model and its metadata."""
    global _model, _metadata

    try:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        if not METADATA_PATH.is_file():
            raise FileNotFoundError(
                f"Model metadata not found: {METADATA_PATH}"
            )

        model = joblib.load(MODEL_PATH)

        with METADATA_PATH.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        feature_names = metadata.get("feature_names")
        if not isinstance(feature_names, list) or not feature_names:
            raise ValueError("Metadata feature_names is missing or invalid")

        if not hasattr(model, "predict_proba"):
            raise ValueError("Model does not support predict_proba")

        if not hasattr(model, "classes_"):
            raise ValueError("Model classes_ is missing")

        if 1 not in list(model.classes_):
            raise ValueError("Model has no phishing class with label 1")

        threshold = float(metadata.get("threshold", 0.5))
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Invalid model threshold")

        _model = model
        _metadata = metadata

        logger.info(
            "Loaded model %s with %d features",
            metadata.get("model_name", "unknown"),
            len(feature_names),
        )
        return True

    except Exception:
        _model = None
        _metadata = None
        logger.exception("Failed to load phishing model")
        return False


def model_is_loaded() -> bool:
    return _model is not None and _metadata is not None


def get_model_info() -> ModelInfoResponse:
    if not model_is_loaded():
        raise RuntimeError("Phishing detection model is unavailable")

    return ModelInfoResponse(
        model_name=str(_metadata.get("model_name", "unknown")),
        feature_count=len(_metadata["feature_names"]),
        feature_names=_metadata["feature_names"],
        threshold=float(_metadata.get("threshold", 0.5)),
        selection_metric=str(
            _metadata.get("selection_metric", "not specified")
        ),
        validation_pr_auc=_metadata.get("validation_pr_auc"),
        probability_calibration=str(
            _metadata.get("probability_calibration", "unknown")
        ),
    )


def _validate_url(url: str) -> str:
    """Validate URL syntax without connecting to the target website."""
    value = url.strip()

    if not value:
        raise ValueError("URL must not be empty")

    if len(value) > 8192:
        raise ValueError("URL exceeds the maximum allowed length")

    if any(ord(char) < 32 for char in value):
        raise ValueError("URL must not contain control characters")

    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("Invalid URL format") from exc

    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Only http:// and https:// URLs are supported")

    if not hostname:
        raise ValueError("URL must contain a hostname")

    if any(char.isspace() for char in hostname):
        raise ValueError("URL hostname must not contain whitespace")

    return value


def _get_phishing_probability(features: dict[str, float]) -> float:
    feature_names = _metadata["feature_names"]

    if set(features) != set(feature_names):
        raise RuntimeError(
            "Extracted feature schema does not match model metadata"
        )

    frame = pd.DataFrame([features], columns=feature_names)
    probabilities = _model.predict_proba(frame)[0]

    classes = list(_model.classes_)
    phishing_index = classes.index(1)
    score = float(probabilities[phishing_index])

    if not math.isfinite(score):
        raise RuntimeError("Model returned an invalid prediction score")

    return min(1.0, max(0.0, score))


def _build_reasons(
    features: dict[str, float],
    prediction: str,
) -> list[str]:
    reasons = []

    if features.get("is_ip", 0.0) == 1.0:
        reasons.append("The hostname is an IP address.")

    if features.get("has_punycode", 0.0) == 1.0:
        reasons.append("The hostname contains punycode.")

    if features.get("has_at_symbol", 0.0) == 1.0:
        reasons.append("The URL contains an @ symbol.")

    if features.get("has_suspicious_keyword", 0.0) == 1.0:
        reasons.append("The URL contains suspicious keywords.")

    if features.get("subdomain_count", 0.0) >= 3.0:
        reasons.append("The hostname contains multiple subdomain levels.")

    if features.get("url_length", 0.0) >= 100.0:
        reasons.append("The URL is unusually long.")

    if not reasons:
        if prediction == "phishing":
            reasons.append(
                "The model detected a phishing-like feature pattern."
            )
        else:
            reasons.append(
                "No listed heuristic indicators were found. "
                "This does not guarantee the website is safe."
            )

    return reasons[:6]


def predict_url(request: PredictionRequest) -> PredictionResponse:
    if not model_is_loaded():
        raise RuntimeError("Phishing detection model is unavailable")

    started = time.perf_counter()
    request_id = str(uuid.uuid4())
    url = _validate_url(request.url)

    try:
        features = extract_features(url)
        score = _get_phishing_probability(features)
        threshold = float(_metadata.get("threshold", 0.5))

        prediction = "phishing" if score >= threshold else "legitimate"

        if prediction == "phishing":
            risk_level = "high" if score >= 0.75 else "medium"
        else:
            risk_level = "low" if score < 0.30 else "medium"

        elapsed_ms = (time.perf_counter() - started) * 1000

        logger.info(
            "Prediction completed request_id=%s prediction=%s "
            "score=%.4f elapsed_ms=%.2f",
            request_id,
            prediction,
            score,
            elapsed_ms,
        )

        return PredictionResponse(
            request_id=request_id,
            url=url,
            prediction=prediction,
            risk_level=risk_level,
            risk_score=round(score, 4),
            threshold=threshold,
            model_name=str(_metadata.get("model_name", "unknown")),
            reasons=_build_reasons(features, prediction),
            processing_time_ms=round(elapsed_ms, 2),
        )

    except ValueError:
        raise
    except Exception:
        logger.exception("Prediction failed request_id=%s", request_id)
        raise
