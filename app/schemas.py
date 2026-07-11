from __future__ import annotations

from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    """One snapshot of machine sensor data. All fields except `type` are
    optional so a client can send a reading with a dropped/missing sensor
    value -- it will be imputed with the training-set median rather than
    rejected outright."""

    air_temperature: float | None = Field(None, description="Air temperature in Kelvin")
    process_temperature: float | None = Field(None, description="Process temperature in Kelvin")
    rotational_speed: float | None = Field(None, description="Rotational speed in rpm")
    torque: float | None = Field(None, description="Torque in Nm")
    tool_wear: float | None = Field(None, description="Tool wear in minutes")
    type: str | None = Field(
        None,
        description=(
            "Product quality variant: L, M, or H (not used by the model yet, "
            "accepted for forward compatibility)"
        ),
    )


class PredictRequest(BaseModel):
    readings: list[SensorReading] = Field(
        ...,
        min_length=1,
        description=(
            "A window of recent sensor readings for one machine, ordered "
            "oldest -> newest. The model was trained on windows of 10 "
            "readings; other lengths are accepted but may reduce accuracy."
        ),
    )


class PredictResponse(BaseModel):
    anomaly_score: float = Field(
        ..., description="Anomaly score in [0, 1], higher = more anomalous"
    )
    is_anomaly: bool = Field(..., description="Whether anomaly_score exceeds the trained threshold")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
