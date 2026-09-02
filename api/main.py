from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.database.database import Base, SessionLocal, engine
from api.database.models import Prediction


app = FastAPI(
    title="Wastewater AI API",
    description="AI-powered wastewater treatment prediction API",
    version="0.1.0",
)


# Create database tables
Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Load trained machine-learning model
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "wastewater_bod5_model.joblib"

model = joblib.load(MODEL_PATH)


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


class PredictionResponse(BaseModel):
    id: int
    predicted_effluent_bod5: float
    unit: str
    status: str
    recommendation: str
    limitations: str


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
    status: str
    created_at: datetime


class PredictionStatsResponse(BaseModel):
    total_predictions: int
    average_predicted_bod5: float
    minimum_predicted_bod5: float
    maximum_predicted_bod5: float


def classify_bod5(predicted_bod5: float) -> str:
    """
    Classify predicted effluent BOD5 into a simple interpretation category.
    """
    if predicted_bod5 <= 20:
        return "low"

    if predicted_bod5 <= 30:
        return "moderate"

    if predicted_bod5 <= 50:
        return "high"

    return "very_high"


def get_treatment_recommendation(predicted_bod5: float) -> str:
    """
    Provide an operational treatment recommendation based on
    the predicted effluent BOD5.

    These recommendations are engineering guidance only and
    are not regulatory compliance determinations.
    """
    status = classify_bod5(predicted_bod5)

    recommendations = {
        "low": (
            "Predicted effluent BOD5 is relatively low. "
            "Continue routine process monitoring and verify "
            "performance with laboratory measurements."
        ),
        "moderate": (
            "Predicted effluent BOD5 is moderate. "
            "Monitor aeration, dissolved oxygen, biological "
            "process conditions, and hydraulic performance."
        ),
        "high": (
            "Predicted effluent BOD5 is elevated. "
            "Review organic loading, aeration, sludge age, "
            "dissolved oxygen, and hydraulic conditions."
        ),
        "very_high": (
            "Predicted effluent BOD5 is very high. "
            "Investigate possible process upset, overloading, "
            "oxygen limitation, inadequate retention time, "
            "or other treatment-process deficiencies."
        ),
    }

    return recommendations[status]


def get_prediction_limitations() -> str:
    """
    Return the general limitations associated with the AI prediction.
    """
    return (
        "This AI prediction is an estimate and should not be used as "
        "a substitute for laboratory testing, engineering judgement, "
        "process monitoring, or applicable regulatory compliance "
        "assessment."
    )


@app.get("/")
def root():
    return {
        "message": "Wastewater AI API is running",
        "version": "0.1.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "wastewater-ai-api",
        "version": "0.1.0",
    }


@app.get("/model")
def model_info():
    return {
        "model_type": "RandomForestRegressor",
        "target": "effluent_bod5",
        "target_unit": "mg/L",
        "features": [
            "influent_bod5",
            "influent_cod",
            "influent_tss",
            "flow_m3_day",
            "dissolved_oxygen",
            "temperature",
            "hrt_hours",
        ],
        "metrics": {
            "mae": 1.40,
            "rmse": 1.48,
            "r2": 0.93,
        },
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: WastewaterInput, db=Depends(get_db)):
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

    predicted_bod5 = round(float(prediction), 2)

    return {
        "id": prediction_record.id,
        "predicted_effluent_bod5": predicted_bod5,
        "unit": "mg/L",
        "status": classify_bod5(predicted_bod5),
        "recommendation": get_treatment_recommendation(predicted_bod5),
        "limitations": get_prediction_limitations(),
    }


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
            "status": classify_bod5(prediction.predicted_effluent_bod5),
            "created_at": prediction.created_at,
        }
        for prediction in predictions
    ]


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

    return {
        "id": prediction.id,
        "influent_bod5": prediction.influent_bod5,
        "influent_cod": prediction.influent_cod,
        "influent_tss": prediction.influent_tss,
        "flow_m3_day": prediction.flow_m3_day,
        "dissolved_oxygen": prediction.dissolved_oxygen,
        "temperature": prediction.temperature,
        "hrt_hours": prediction.hrt_hours,
        "predicted_effluent_bod5": prediction.predicted_effluent_bod5,
        "status": classify_bod5(prediction.predicted_effluent_bod5),
        "created_at": prediction.created_at,
    }