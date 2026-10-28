import logging

from fastapi import APIRouter, HTTPException, status

from app.schemas.prediction import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from app.services.prediction_service import (
    get_model_info,
    model_is_loaded,
    predict_url,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    loaded = model_is_loaded()
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
    )


@router.get(
    "/api/v1/model-info",
    response_model=ModelInfoResponse,
)
def model_info() -> ModelInfoResponse:
    try:
        return get_model_info()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Phishing detection model is unavailable",
        ) from exc


@router.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        return predict_url(request)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        logger.error("Prediction service unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Phishing detection model is unavailable",
        ) from exc
    except Exception as exc:
        logger.exception("Unhandled prediction endpoint error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed",
        ) from exc
