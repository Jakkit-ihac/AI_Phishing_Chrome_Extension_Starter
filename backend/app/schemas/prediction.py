from typing import Literal

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    url: str = Field(
        ...,
        min_length=1,
        max_length=8192,
        description="URL to analyze for phishing indicators",
        examples=["https://example.com"],
    )


class PredictionResponse(BaseModel):
    request_id: str
    url: str
    prediction: Literal["legitimate", "phishing"]
    risk_level: Literal["low", "medium", "high"]
    risk_score: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    model_name: str
    reasons: list[str]
    processing_time_ms: float = Field(ge=0.0)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    feature_count: int
    feature_names: list[str]
    threshold: float
    selection_metric: str
    validation_pr_auc: float | None = None
    probability_calibration: str
