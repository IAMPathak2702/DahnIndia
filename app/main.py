from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.model import AnomalyModel
from app.schemas import HealthResponse, PredictRequest, PredictResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("iot-anomaly-service")

# Populated once in the lifespan handler below, read (never mutated) by
# request handlers. This is what "load the model once at startup, not per
# request" means in practice with FastAPI.
model_state: dict[str, AnomalyModel] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Loading anomaly model artifact...")
    model_state["model"] = AnomalyModel()
    logger.info(
        "Model loaded. window_size=%s threshold=%.3f",
        model_state["model"].window_size,
        model_state["model"].threshold,
    )
    yield
    model_state.clear()


app = FastAPI(
    title="IoT Anomaly Detection Service",
    description="Flags abnormal machine sensor windows as an early sign of possible failure.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded="model" in model_state)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    model = model_state.get("model")
    if model is None:
        # Should only happen if a request somehow lands before startup finishes.
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    readings = [r.model_dump() for r in request.readings]
    try:
        anomaly_score, is_anomaly = model.score(readings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PredictResponse(anomaly_score=anomaly_score, is_anomaly=is_anomaly)
