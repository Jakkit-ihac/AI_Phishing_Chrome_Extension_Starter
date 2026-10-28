import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.services.prediction_service import load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="AI Phishing Detection API",
    description=(
        "Detect phishing-like URLs using a locally trained "
        "machine learning model. The API does not visit submitted URLs."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
