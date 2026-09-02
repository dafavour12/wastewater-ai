from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.database.database import Base, SessionLocal, engine
from api.database.models import Prediction


# --------------------------------------------------
# Application setup
# --------------------------------------------------

app = FastAPI(
    title="Wastewater AI API",
    description="AI-powered wastewater treatment prediction API",
    version="0.1.0",
)


# Create database tables
Base.metadata.create_all(bind=engine)


# --------------------------------------------------
# Database dependency
# --------------------------------------------------

def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


# --------------------------------------------------
# Load trained ML model
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "wastewater_bod5_model.joblib"

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# Input schema
# --------------------------------------------------

class WastewaterInput(BaseModel):
    influent_bod5: float = Field(
        ...,
        ge=0,
        description="Influent BOD5 in mg/L",
    )

    influent_cod: float = Field(
        ...,
        ge=0,
        description="Influent COD in mg/L",
    )

    influent_tss: float = Field(
        ...,
        ge=0,
        description="Influent TSS in mg/L",
    )

    flow_m3_day: float = Field(
        ...,
        gt=0,
        description="Wastewater flow rate in m³/day",
    )

    dissolved_oxygen: float = Field(
        ...,
        ge=0,
        description="Dissolved oxygen in mg/L",
    )

    temperature: float = Field(
        ...,
        ge=0,
        le=60,
        description="Wastewater temperature in °C",
    )

    hrt_hours: float = Field(
        ...,
        gt=0,
        description="Hydraulic retention time in hours",
    )


# --------------------------------------------------
# Response schemas
# --------------------------------------------------

class PredictionResponse(BaseModel):
    id: int
    predicted_effluent_bod5: float
    unit: str


class PredictionHistoryResponse(BaseModel):
    id: int
    influent_bod5: float
    influent_cod: float
    influent_tss: float
    flow_m3_day: float
    dissolved_oxygen: float
    temperature: float
    hrt_hours: float
    predicted_effluent_bod5: float
    created_at: datetime


class PredictionStatsResponse(BaseModel):
    total_predictions: int
    average_predicted_bod5: float
    minimum_predicted_bod5: float
    maximum_predicted_bod5: float


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Wastewater AI API is running",
        "version": "0.1.0",
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    data: WastewaterInput,
    db=Depends(get_db),
):

    features = pd.DataFrame(
        [[
            data.influent_bod5,
            data.influent_cod,
            data.influent_tss,
            data.flow_m3_day,
            data.dissolved_oxygen,
            data.temperature,
            data.hrt_hours,
        ]],
        columns=[
            "influent_bod5",
            "influent_cod",
            "influent_tss",
            "flow_m3_day",
            "dissolved_oxygen",
            "temperature",
            "hrt_hours",
        ],
    )

    prediction = model.predict(features)[0]

    prediction_record = Prediction(
        influent_bod5=data.influent_bod5,
        influent_cod=data.influent_cod,
        influent_tss=data.influent_tss,
        flow_m3_day=data.flow_m3_day,
        dissolved_oxygen=data.dissolved_oxygen,
        temperature=data.temperature,
        hrt_hours=data.hrt_hours,
        predicted_effluent_bod5=float(prediction),
    )

    db.add(prediction_record)
    db.commit()
    db.refresh(prediction_record)

    return {
        "id": prediction_record.id,
        "predicted_effluent_bod5": round(
            float(prediction),
            2,
        ),
        "unit": "mg/L",
    }

# --------------------------------------------------
# Prediction history endpoint
# --------------------------------------------------

@app.get(
    "/predictions",
    response_model=list[PredictionHistoryResponse],
)
def get_predictions(db=Depends(get_db)):

    predictions = (
        db.query(Prediction)
        .order_by(Prediction.created_at.desc())
        .all()
    )

    return [
        {
            "id": prediction.id,
            "influent_bod5": prediction.influent_bod5,
            "influent_cod": prediction.influent_cod,
            "influent_tss": prediction.influent_tss,
            "flow_m3_day": prediction.flow_m3_day,
            "dissolved_oxygen": prediction.dissolved_oxygen,
            "temperature": prediction.temperature,
            "hrt_hours": prediction.hrt_hours,
            "predicted_effluent_bod5": prediction.predicted_effluent_bod5,
            "created_at": prediction.created_at,
        }
        for prediction in predictions
    ]


# --------------------------------------------------
# Prediction statistics endpoint
# --------------------------------------------------

@app.get(
    "/predictions/stats",
    response_model=PredictionStatsResponse,
)
def get_prediction_stats(db=Depends(get_db)):

    predictions = db.query(Prediction).all()

    if not predictions:
        return {
            "total_predictions": 0,
            "average_predicted_bod5": 0.0,
            "minimum_predicted_bod5": 0.0,
            "maximum_predicted_bod5": 0.0,
        }

    values = [
        prediction.predicted_effluent_bod5
        for prediction in predictions
    ]

    return {
        "total_predictions": len(values),
        "average_predicted_bod5": round(
            sum(values) / len(values),
            2,
        ),
        "minimum_predicted_bod5": round(
            min(values),
            2,
        ),
        "maximum_predicted_bod5": round(
            max(values),
            2,
        ),
    }


# --------------------------------------------------
# Single prediction endpoint
# --------------------------------------------------

@app.get(
    "/predictions/{prediction_id}",
    response_model=PredictionHistoryResponse,
)
def get_prediction(
    prediction_id: int,
    db=Depends(get_db),
):

    prediction = (
        db.query(Prediction)
        .filter(Prediction.id == prediction_id)
        .first()
    )

    if prediction is None:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found",
        )

    return prediction