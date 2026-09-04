from datetime import datetime
from pathlib import Path
import math

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field

from api.config import (
    get_bod5_model_path,
    get_environment,
    get_process_model_path,
)
from api.database.database import Base, SessionLocal, engine
from api.database.migrations import (
    migrate_risk_assessment_decision_fields,
)
from api.database.models import Prediction, RiskAssessmentRecord
from api.errors import (
    http_exception_handler,
    unexpected_exception_handler,
    validation_exception_handler,
)
from api.guardrails import validate_prediction_guardrails
from decision.engine import generate_decision
from decision.models import DecisionInput, DecisionRecommendation
from process.monitoring import PROCESS_FEATURES, ProcessMonitor
from risk.scoring import RiskAssessment, assess_risk


app = FastAPI(
    title="Wastewater AI API",
    description=(
        "AI-powered wastewater treatment prediction and "
        "risk assessment API"
    ),
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------

app.add_exception_handler(
    HTTPException,
    http_exception_handler,
)

app.add_exception_handler(
    RequestValidationError,
    validation_exception_handler,
)

app.add_exception_handler(
    Exception,
    unexpected_exception_handler,
)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

# Create any tables that do not already exist.
#
# IMPORTANT:
# SQLAlchemy's create_all() does NOT modify existing tables.
# Therefore, it cannot add the new V2.7 Decision Engine columns
# to an existing wastewater.db.
Base.metadata.create_all(bind=engine)


# Run the V2.7 risk-assessment migration after table creation.
#
# This is safe for both:
# - a brand-new database, where all columns already exist
# - an existing wastewater.db, where only missing columns are added
#
# Existing records are preserved.
migrated_columns = migrate_risk_assessment_decision_fields(
    engine
)


def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent


def resolve_model_path(configured_path: str) -> Path:
    path = Path(configured_path)

    if path.is_absolute():
        return path

    return BASE_DIR / path


MODEL_PATH = resolve_model_path(
    get_bod5_model_path()
)

PROCESS_MODEL_PATH = resolve_model_path(
    get_process_model_path()
)


# ---------------------------------------------------------------------------
# Safe model loading
# ---------------------------------------------------------------------------

model = None
model_error: str | None = None

process_monitor = None
process_model_error: str | None = None


def load_bod5_model():
    """
    Load the BOD5 prediction model without allowing a model-loading
    failure to crash the API during import/startup.
    """
    global model
    global model_error

    try:
        model = joblib.load(MODEL_PATH)
        model_error = None
        return model

    except FileNotFoundError:
        model = None
        model_error = "BOD5 prediction model file was not found."
        return None

    except (OSError, ValueError, EOFError) as exc:
        model = None
        model_error = (
            "BOD5 prediction model could not be loaded."
        )

        _ = exc

        return None

    except Exception as exc:
        model = None
        model_error = (
            "BOD5 prediction model could not be loaded."
        )

        _ = exc

        return None


def load_process_model():
    """
    Load the process anomaly model without allowing a loading failure
    to crash the API during import/startup.
    """
    global process_monitor
    global process_model_error

    try:
        process_monitor = ProcessMonitor(
            model_path=PROCESS_MODEL_PATH,
        )

        process_model_error = None
        return process_monitor

    except FileNotFoundError:
        process_monitor = None
        process_model_error = (
            "Process anomaly model file was not found."
        )
        return None

    except (OSError, ValueError, EOFError) as exc:
        process_monitor = None
        process_model_error = (
            "Process anomaly model could not be loaded."
        )

        _ = exc

        return None

    except Exception as exc:
        process_monitor = None
        process_model_error = (
            "Process anomaly model could not be loaded."
        )

        _ = exc

        return None


load_bod5_model()
load_process_model()


# ---------------------------------------------------------------------------
# Model runtime helpers
# ---------------------------------------------------------------------------

def require_bod5_model():
    """
    Return the loaded BOD5 model or raise a controlled 503 error.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="BOD5 prediction model is currently unavailable.",
        )

    return model


def require_process_monitor():
    """
    Return the loaded process monitor or raise a controlled 503 error.
    """
    if process_monitor is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Process anomaly model is currently unavailable."
            ),
        )

    return process_monitor


def predict_bod5(
    features: pd.DataFrame,
) -> float:
    """
    Execute BOD5 model inference safely.

    Model failures are converted into a controlled application-level
    error instead of allowing raw model exceptions to propagate.
    """
    prediction_model = require_bod5_model()

    try:
        prediction = prediction_model.predict(
            features
        )[0]

        predicted_bod5 = float(prediction)

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "BOD5 prediction could not be completed because "
                "the prediction model failed during inference."
            ),
        ) from exc

    if not math.isfinite(predicted_bod5):
        raise HTTPException(
            status_code=503,
            detail=(
                "BOD5 prediction could not be completed because "
                "the model returned a non-finite result."
            ),
        )

    return predicted_bod5


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class StatusResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str
    models_ready: bool


class ModelStatusResponse(BaseModel):
    name: str
    available: bool
    path: str
    model_type: str


class ModelsStatusResponse(BaseModel):
    status: str
    models: list[ModelStatusResponse]


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


class AnalyticsResponse(BaseModel):
    total_predictions: int
    average_predicted_bod5: float
    minimum_predicted_bod5: float
    maximum_predicted_bod5: float
    status_counts: dict[str, int]
    recent_predictions: list[PredictionHistoryResponse]


class RiskAssessmentInput(BaseModel):
    predicted_bod5: float = Field(
        ...,
        ge=0,
        description="Predicted effluent BOD5 in mg/L",
    )

    anomaly_percentile: float = Field(
        ...,
        ge=0,
        le=100,
        description=(
            "Process anomaly percentile from the "
            "process-monitoring engine (0-100)"
        ),
    )

    model_confidence: str = Field(
        ...,
        description=(
            "Confidence category: high, medium, moderate, "
            "low, limited, or research"
        ),
    )


class RiskAssessmentResponse(BaseModel):
    risk_level: str
    risk_score: float
    prediction_score: float
    anomaly_score: float
    confidence_score: float
    predicted_bod5: float
    anomaly_percentile: float
    risk_reason: str
    recommended_action: str


class ProcessAnomalyInput(BaseModel):
    PH_P: float = Field(
        ...,
        alias="PH-P",
        ge=0,
        description="Process pH",
    )

    DBO_P: float = Field(
        ...,
        alias="DBO-P",
        ge=0,
        description="Primary process BOD5",
    )

    SS_P: float = Field(
        ...,
        alias="SS-P",
        ge=0,
        description="Primary process suspended solids",
    )

    SSV_P: float = Field(
        ...,
        alias="SSV-P",
        ge=0,
        description="Primary process volatile suspended solids",
    )

    SED_P: float = Field(
        ...,
        alias="SED-P",
        ge=0,
        description="Primary process sedimentation",
    )

    COND_P: float = Field(
        ...,
        alias="COND-P",
        ge=0,
        description="Primary process conductivity",
    )

    PH_D: float = Field(
        ...,
        alias="PH-D",
        ge=0,
        description="Secondary process pH",
    )

    DBO_D: float = Field(
        ...,
        alias="DBO-D",
        ge=0,
        description="Secondary process BOD5",
    )

    DQO_D: float = Field(
        ...,
        alias="DQO-D",
        ge=0,
        description="Secondary process COD",
    )

    SS_D: float = Field(
        ...,
        alias="SS-D",
        ge=0,
        description="Secondary process suspended solids",
    )

    SSV_D: float = Field(
        ...,
        alias="SSV-D",
        ge=0,
        description="Secondary process volatile suspended solids",
    )

    SED_D: float = Field(
        ...,
        alias="SED-D",
        ge=0,
        description="Secondary process sedimentation",
    )

    COND_D: float = Field(
        ...,
        alias="COND-D",
        ge=0,
        description="Secondary process conductivity",
    )

    RD_DBO_P: float = Field(
        ...,
        alias="RD-DBO-P",
        ge=0,
        description="Primary BOD reduction ratio",
    )

    RD_SS_P: float = Field(
        ...,
        alias="RD-SS-P",
        ge=0,
        description="Primary suspended solids reduction ratio",
    )

    RD_DBO_D: float = Field(
        ...,
        alias="RD-DBO-D",
        ge=0,
        description="Secondary BOD reduction ratio",
    )

    RD_SS_D: float = Field(
        ...,
        alias="RD-SS-D",
        ge=0,
        description="Secondary suspended solids reduction ratio",
    )

    RD_DBO_G: float = Field(
        ...,
        alias="RD-DBO-G",
        ge=0,
        description="Global BOD reduction ratio",
    )

    RD_SS_G: float = Field(
        ...,
        alias="RD-SS-G",
        ge=0,
        description="Global suspended solids reduction ratio",
    )

    RD_SED_G: float = Field(
        ...,
        alias="RD-SED-G",
        ge=0,
        description="Global sedimentation reduction ratio",
    )

    RD_N_NH4: float = Field(
        ...,
        alias="RD-N-NH4",
        ge=0,
        description="Ammonium nitrogen reduction ratio",
    )

    RD_N_NO2: float = Field(
        ...,
        alias="RD-N-NO2",
        ge=0,
        description="Nitrite nitrogen reduction ratio",
    )

    model_config = {
        "populate_by_name": True,
    }


class ProcessAnomalyResponse(BaseModel):
    anomaly_score: float
    anomaly_percentile: float
    is_anomaly: bool
    risk_band: str
    alert_level: str
    message: str
    monitoring_method: str
    contamination: float
    features_used: int


class CombinedRiskAssessmentInput(BaseModel):
    wastewater: WastewaterInput
    process: ProcessAnomalyInput

    model_confidence: str = Field(
        ...,
        description=(
            "Confidence category: high, medium, moderate, "
            "low, limited, or research"
        ),
    )


class CombinedRiskAssessmentResponse(BaseModel):
    assessment_id: int
    predicted_effluent_bod5: float
    prediction_status: str
    anomaly_score: float
    anomaly_percentile: float
    is_anomaly: bool
    anomaly_risk_band: str
    anomaly_alert_level: str
    overall_risk_level: str
    overall_risk_score: float
    prediction_score: float
    confidence_score: float
    risk_reason: str
    recommended_action: str
    monitoring_method: str
    contamination: float
    process_features_used: int

    # V2.7 Decision Engine output
    decision_priority: str
    decision_summary: str
    possible_contributors: list[str]
    checks_to_perform: list[str]
    recommended_actions: list[str]
    monitoring_recommendations: list[str]
    evidence: list[str]
    limitations: list[str]


class RiskAssessmentHistoryResponse(BaseModel):
    id: int
    predicted_effluent_bod5: float
    prediction_status: str
    anomaly_score: float
    anomaly_percentile: float
    is_anomaly: bool
    anomaly_risk_band: str
    anomaly_alert_level: str
    overall_risk_level: str
    overall_risk_score: float
    prediction_score: float
    confidence_score: float
    model_confidence: str
    risk_reason: str
    recommended_action: str
    monitoring_method: str
    contamination: float
    process_features_used: int

    # V2.7 Decision Engine output.
    #
    # These are nullable because older records were created before
    # the Decision Engine fields existed.
    decision_priority: str | None = None
    decision_summary: str | None = None
    possible_contributors: list[str] | None = None
    checks_to_perform: list[str] | None = None
    recommended_actions: list[str] | None = None
    monitoring_recommendations: list[str] | None = None
    evidence: list[str] | None = None
    limitations: list[str] | None = None

    created_at: datetime


class RiskAssessmentStatsResponse(BaseModel):
    total_assessments: int
    average_risk_score: float
    minimum_risk_score: float
    maximum_risk_score: float
    risk_level_counts: dict[str, int]
    anomaly_count: int


# ---------------------------------------------------------------------------
# V2.7 Decision and recommendation API models
# ---------------------------------------------------------------------------

class DecisionRecommendationInput(BaseModel):
    predicted_effluent_bod5: float = Field(
        ...,
        ge=0,
        description="Predicted effluent BOD5 in mg/L",
    )

    anomaly_percentile: float = Field(
        ...,
        ge=0,
        le=100,
        description="Process anomaly percentile from 0 to 100",
    )

    overall_risk_level: int = Field(
        ...,
        ge=0,
        description=(
            "Numeric overall risk level: "
            "0 normal, 1 low, 2 elevated, 3 high, 4 critical"
        ),
    )

    overall_risk_score: float = Field(
        ...,
        ge=0,
        description="Combined risk score",
    )

    dissolved_oxygen: float = Field(
        ...,
        ge=0,
        description="Dissolved oxygen in mg/L",
    )

    flow_m3_day: float = Field(
        ...,
        gt=0,
        description="Wastewater flow rate in m³/day",
    )

    hrt_hours: float = Field(
        ...,
        gt=0,
        description="Hydraulic retention time in hours",
    )

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

    model_confidence: str = Field(
        ...,
        description=(
            "Model confidence category: high, medium, moderate, "
            "low, limited, research, or unknown"
        ),
    )


class DecisionRecommendationResponse(BaseModel):
    priority: str
    summary: str
    possible_contributors: list[str]
    checks_to_perform: list[str]
    recommended_actions: list[str]
    monitoring_recommendations: list[str]
    evidence: list[str]
    limitations: list[str]


# ---------------------------------------------------------------------------
# BOD5 classification and recommendations
# ---------------------------------------------------------------------------

def classify_bod5(predicted_bod5: float) -> str:
    if predicted_bod5 <= 20:
        return "low"

    if predicted_bod5 <= 30:
        return "moderate"

    if predicted_bod5 <= 50:
        return "high"

    return "very_high"


def get_treatment_recommendation(
    predicted_bod5: float,
) -> str:
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
    return (
        "This AI prediction is an estimate and should not be used as "
        "a substitute for laboratory testing, engineering judgement, "
        "process monitoring, or applicable regulatory compliance "
        "assessment."
    )


# ---------------------------------------------------------------------------
# Basic endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Wastewater AI API is running",
        "version": "0.1.0",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health():
    return {
        "status": "healthy",
        "service": "wastewater-ai-api",
        "version": "0.1.0",
    }


@app.get("/status", response_model=StatusResponse)
def status():
    bod5_available = model is not None
    process_available = process_monitor is not None

    models_ready = (
        bod5_available
        and process_available
    )

    return {
        "status": "ready" if models_ready else "degraded",
        "service": "wastewater-ai-api",
        "version": "0.1.0",
        "environment": get_environment(),
        "models_ready": models_ready,
    }


@app.get(
    "/models/status",
    response_model=ModelsStatusResponse,
)
def models_status():
    bod5_available = model is not None
    process_available = process_monitor is not None

    all_available = (
        bod5_available
        and process_available
    )

    return {
        "status": "ready" if all_available else "degraded",
        "models": [
            {
                "name": "BOD5 prediction model",
                "available": bod5_available,
                "path": str(MODEL_PATH),
                "model_type": "RandomForestRegressor",
            },
            {
                "name": "Process anomaly model",
                "available": process_available,
                "path": str(PROCESS_MODEL_PATH),
                "model_type": "IsolationForest",
            },
        ],
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


# ---------------------------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    data: WastewaterInput,
    db=Depends(get_db),
):
    guardrail_data = {
        "influent_bod5": data.influent_bod5,
        "influent_cod": data.influent_cod,
        "influent_tss": data.influent_tss,
        "flow_m3_day": data.flow_m3_day,
        "dissolved_oxygen": data.dissolved_oxygen,
        "temperature": data.temperature,
        "hrt_hours": data.hrt_hours,
    }

    violations = validate_prediction_guardrails(
        guardrail_data
    )

    if violations:
        raise HTTPException(
            status_code=422,
            detail=[
                {
                    "field": violation.field,
                    "value": violation.value,
                    "message": violation.message,
                }
                for violation in violations
            ],
        )

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

    prediction = predict_bod5(
        features
    )

    prediction_record = Prediction(
        influent_bod5=data.influent_bod5,
        influent_cod=data.influent_cod,
        influent_tss=data.influent_tss,
        flow_m3_day=data.flow_m3_day,
        dissolved_oxygen=data.dissolved_oxygen,
        temperature=data.temperature,
        hrt_hours=data.hrt_hours,
        predicted_effluent_bod5=prediction,
    )

    try:
        db.add(prediction_record)
        db.commit()
        db.refresh(prediction_record)

    except Exception as exc:
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction was calculated, but the result "
                "could not be saved."
            ),
        ) from exc

    predicted_bod5 = round(
        prediction,
        2,
    )

    return {
        "id": prediction_record.id,
        "predicted_effluent_bod5": predicted_bod5,
        "unit": "mg/L",
        "status": classify_bod5(
            predicted_bod5
        ),
        "recommendation": get_treatment_recommendation(
            predicted_bod5
        ),
        "limitations": get_prediction_limitations(),
    }


# ---------------------------------------------------------------------------
# Process anomaly endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/process/anomaly",
    response_model=ProcessAnomalyResponse,
)
def process_anomaly(
    data: ProcessAnomalyInput,
):
    monitor = require_process_monitor()

    process_data = {
        feature: getattr(
            data,
            feature.replace("-", "_"),
        )
        for feature in PROCESS_FEATURES
    }

    try:
        result = monitor.predict(
            process_data
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Process anomaly detection could not be completed "
                "because the anomaly model failed during inference."
            ),
        ) from exc

    return {
        "anomaly_score": round(
            result.anomaly_score,
            4,
        ),
        "anomaly_percentile": round(
            result.anomaly_percentile,
            2,
        ),
        "is_anomaly": result.is_anomaly,
        "risk_band": result.risk_band,
        "alert_level": result.alert_level,
        "message": result.message,
        "monitoring_method": "Isolation Forest",
        "contamination": monitor.contamination,
        "features_used": len(PROCESS_FEATURES),
    }


# ---------------------------------------------------------------------------
# Standalone risk assessment
# ---------------------------------------------------------------------------

@app.post(
    "/risk/assess",
    response_model=RiskAssessmentResponse,
)
def assess_risk_endpoint(
    data: RiskAssessmentInput,
):
    try:
        assessment: RiskAssessment = assess_risk(
            predicted_bod5=data.predicted_bod5,
            anomaly_percentile=data.anomaly_percentile,
            model_confidence=data.model_confidence,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    return {
        "risk_level": assessment.risk_level.name.lower(),
        "risk_score": round(
            assessment.risk_score,
            2,
        ),
        "prediction_score": round(
            assessment.prediction_score,
            2,
        ),
        "anomaly_score": round(
            assessment.anomaly_score,
            2,
        ),
        "confidence_score": round(
            assessment.confidence_score,
            2,
        ),
        "predicted_bod5": round(
            assessment.predicted_bod5,
            2,
        ),
        "anomaly_percentile": round(
            assessment.anomaly_percentile,
            2,
        ),
        "risk_reason": assessment.risk_reason,
        "recommended_action": assessment.recommended_action,
    }


# ---------------------------------------------------------------------------
# Combined risk assessment
# ---------------------------------------------------------------------------

@app.post(
    "/risk/assess/process",
    response_model=CombinedRiskAssessmentResponse,
)
def assess_process_risk(
    data: CombinedRiskAssessmentInput,
    db=Depends(get_db),
):
    guardrail_data = {
        "influent_bod5": data.wastewater.influent_bod5,
        "influent_cod": data.wastewater.influent_cod,
        "influent_tss": data.wastewater.influent_tss,
        "flow_m3_day": data.wastewater.flow_m3_day,
        "dissolved_oxygen": data.wastewater.dissolved_oxygen,
        "temperature": data.wastewater.temperature,
        "hrt_hours": data.wastewater.hrt_hours,
    }

    violations = validate_prediction_guardrails(
        guardrail_data
    )

    if violations:
        raise HTTPException(
            status_code=422,
            detail=[
                {
                    "field": violation.field,
                    "value": violation.value,
                    "message": violation.message,
                }
                for violation in violations
            ],
        )

    features = pd.DataFrame(
        [[
            data.wastewater.influent_bod5,
            data.wastewater.influent_cod,
            data.wastewater.influent_tss,
            data.wastewater.flow_m3_day,
            data.wastewater.dissolved_oxygen,
            data.wastewater.temperature,
            data.wastewater.hrt_hours,
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

    prediction = predict_bod5(
        features
    )

    predicted_bod5 = round(
        prediction,
        2,
    )

    monitor = require_process_monitor()

    process_data = {
        feature: getattr(
            data.process,
            feature.replace("-", "_"),
        )
        for feature in PROCESS_FEATURES
    }

    try:
        process_result = monitor.predict(
            process_data
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Process anomaly detection could not be completed "
                "because the anomaly model failed during inference."
            ),
        ) from exc

    try:
        assessment: RiskAssessment = assess_risk(
            predicted_bod5=predicted_bod5,
            anomaly_percentile=(
                process_result.anomaly_percentile
            ),
            model_confidence=data.model_confidence,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    # -----------------------------------------------------------------------
    # V2.7 Decision Engine integration
    # -----------------------------------------------------------------------
    #
    # The Decision Engine receives the outputs of the prediction,
    # anomaly-detection, and combined-risk layers together with the
    # operating conditions supplied by the user.
    decision_input = DecisionInput(
        predicted_effluent_bod5=predicted_bod5,
        anomaly_percentile=(
            process_result.anomaly_percentile
        ),
        overall_risk_level=int(
            assessment.risk_level
        ),
        overall_risk_score=assessment.risk_score,
        dissolved_oxygen=data.wastewater.dissolved_oxygen,
        flow_m3_day=data.wastewater.flow_m3_day,
        hrt_hours=data.wastewater.hrt_hours,
        influent_bod5=data.wastewater.influent_bod5,
        influent_cod=data.wastewater.influent_cod,
        influent_tss=data.wastewater.influent_tss,
        model_confidence=data.model_confidence,
    )

    try:
        decision: DecisionRecommendation = generate_decision(
            decision_input
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    risk_level = assessment.risk_level.name.lower()

    # -----------------------------------------------------------------------
    # Persist combined risk + Decision Engine result
    # -----------------------------------------------------------------------
    risk_record = RiskAssessmentRecord(
        predicted_effluent_bod5=predicted_bod5,
        prediction_status=classify_bod5(
            predicted_bod5
        ),
        anomaly_score=round(
            process_result.anomaly_score,
            4,
        ),
        anomaly_percentile=round(
            process_result.anomaly_percentile,
            2,
        ),
        is_anomaly=process_result.is_anomaly,
        anomaly_risk_band=process_result.risk_band,
        anomaly_alert_level=process_result.alert_level,
        overall_risk_level=risk_level,
        overall_risk_score=round(
            assessment.risk_score,
            2,
        ),
        prediction_score=round(
            assessment.prediction_score,
            2,
        ),
        confidence_score=round(
            assessment.confidence_score,
            2,
        ),
        model_confidence=data.model_confidence,
        risk_reason=assessment.risk_reason,
        recommended_action=assessment.recommended_action,
        monitoring_method="Isolation Forest",
        contamination=monitor.contamination,
        process_features_used=len(PROCESS_FEATURES),

        # V2.7 Decision Engine fields
        decision_priority=decision.priority.name.lower(),
        decision_summary=decision.summary,
        possible_contributors=decision.possible_contributors,
        checks_to_perform=decision.checks_to_perform,
        recommended_actions=decision.recommended_actions,
        monitoring_recommendations=(
            decision.monitoring_recommendations
        ),
        evidence=decision.evidence,
        limitations=decision.limitations,
    )

    try:
        db.add(risk_record)
        db.commit()
        db.refresh(risk_record)

    except Exception as exc:
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=(
                "Risk assessment was calculated, but the result "
                "could not be saved."
            ),
        ) from exc

    return {
        "assessment_id": risk_record.id,
        "predicted_effluent_bod5": predicted_bod5,
        "prediction_status": classify_bod5(
            predicted_bod5
        ),
        "anomaly_score": round(
            process_result.anomaly_score,
            4,
        ),
        "anomaly_percentile": round(
            process_result.anomaly_percentile,
            2,
        ),
        "is_anomaly": process_result.is_anomaly,
        "anomaly_risk_band": process_result.risk_band,
        "anomaly_alert_level": process_result.alert_level,
        "overall_risk_level": risk_level,
        "overall_risk_score": round(
            assessment.risk_score,
            2,
        ),
        "prediction_score": round(
            assessment.prediction_score,
            2,
        ),
        "confidence_score": round(
            assessment.confidence_score,
            2,
        ),
        "risk_reason": assessment.risk_reason,
        "recommended_action": assessment.recommended_action,
        "monitoring_method": "Isolation Forest",
        "contamination": monitor.contamination,
        "process_features_used": len(PROCESS_FEATURES),

        # V2.7 Decision Engine output
        "decision_priority": decision.priority.name.lower(),
        "decision_summary": decision.summary,
        "possible_contributors": (
            decision.possible_contributors
        ),
        "checks_to_perform": (
            decision.checks_to_perform
        ),
        "recommended_actions": (
            decision.recommended_actions
        ),
        "monitoring_recommendations": (
            decision.monitoring_recommendations
        ),
        "evidence": decision.evidence,
        "limitations": decision.limitations,
    }


# ---------------------------------------------------------------------------
# V2.7 Decision and recommendation endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/decision/recommend",
    response_model=DecisionRecommendationResponse,
)
def decision_recommendation(
    data: DecisionRecommendationInput,
):
    """
    Generate a deterministic engineering recommendation from
    prediction, anomaly, risk, and operating-condition indicators.

    This endpoint does not make a root-cause diagnosis.
    It identifies possible contributors and recommended checks
    based on transparent engineering rules.
    """

    decision_input = DecisionInput(
        predicted_effluent_bod5=data.predicted_effluent_bod5,
        anomaly_percentile=data.anomaly_percentile,
        overall_risk_level=data.overall_risk_level,
        overall_risk_score=data.overall_risk_score,
        dissolved_oxygen=data.dissolved_oxygen,
        flow_m3_day=data.flow_m3_day,
        hrt_hours=data.hrt_hours,
        influent_bod5=data.influent_bod5,
        influent_cod=data.influent_cod,
        influent_tss=data.influent_tss,
        model_confidence=data.model_confidence,
    )

    try:
        decision: DecisionRecommendation = generate_decision(
            decision_input
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    return {
        "priority": decision.priority.name.lower(),
        "summary": decision.summary,
        "possible_contributors": (
            decision.possible_contributors
        ),
        "checks_to_perform": (
            decision.checks_to_perform
        ),
        "recommended_actions": (
            decision.recommended_actions
        ),
        "monitoring_recommendations": (
            decision.monitoring_recommendations
        ),
        "evidence": decision.evidence,
        "limitations": decision.limitations,
    }


# ---------------------------------------------------------------------------
# Risk assessment history
# ---------------------------------------------------------------------------

@app.get(
    "/risk/assessments",
    response_model=list[RiskAssessmentHistoryResponse],
)
def get_risk_assessments(
    db=Depends(get_db),
):
    assessments = (
        db.query(RiskAssessmentRecord)
        .order_by(
            RiskAssessmentRecord.created_at.desc()
        )
        .all()
    )

    return assessments


@app.get(
    "/risk/assessments/stats",
    response_model=RiskAssessmentStatsResponse,
)
def get_risk_assessment_stats(
    db=Depends(get_db),
):
    assessments = (
        db.query(RiskAssessmentRecord)
        .all()
    )

    if not assessments:
        return {
            "total_assessments": 0,
            "average_risk_score": 0.0,
            "minimum_risk_score": 0.0,
            "maximum_risk_score": 0.0,
            "risk_level_counts": {
                "normal": 0,
                "low": 0,
                "elevated": 0,
                "high": 0,
                "critical": 0,
            },
            "anomaly_count": 0,
        }

    scores = [
        assessment.overall_risk_score
        for assessment in assessments
    ]

    risk_level_counts = {
        "normal": 0,
        "low": 0,
        "elevated": 0,
        "high": 0,
        "critical": 0,
    }

    for assessment in assessments:
        level = assessment.overall_risk_level

        if level in risk_level_counts:
            risk_level_counts[level] += 1

    anomaly_count = sum(
        1
        for assessment in assessments
        if assessment.is_anomaly
    )

    return {
        "total_assessments": len(assessments),
        "average_risk_score": round(
            sum(scores) / len(scores),
            2,
        ),
        "minimum_risk_score": round(
            min(scores),
            2,
        ),
        "maximum_risk_score": round(
            max(scores),
            2,
        ),
        "risk_level_counts": risk_level_counts,
        "anomaly_count": anomaly_count,
    }


@app.get(
    "/risk/assessments/{assessment_id}",
    response_model=RiskAssessmentHistoryResponse,
)
def get_risk_assessment(
    assessment_id: int,
    db=Depends(get_db),
):
    assessment = (
        db.query(RiskAssessmentRecord)
        .filter(
            RiskAssessmentRecord.id == assessment_id
        )
        .first()
    )

    if assessment is None:
        raise HTTPException(
            status_code=404,
            detail="Risk assessment not found",
        )

    return assessment


# ---------------------------------------------------------------------------
# Prediction history
# ---------------------------------------------------------------------------

@app.get(
    "/predictions",
    response_model=list[PredictionHistoryResponse],
)
def get_predictions(
    db=Depends(get_db),
):
    predictions = (
        db.query(Prediction)
        .order_by(
            Prediction.created_at.desc()
        )
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
            "predicted_effluent_bod5": (
                prediction.predicted_effluent_bod5
            ),
            "status": classify_bod5(
                prediction.predicted_effluent_bod5
            ),
            "created_at": prediction.created_at,
        }
        for prediction in predictions
    ]


@app.get(
    "/predictions/stats",
    response_model=PredictionStatsResponse,
)
def get_prediction_stats(
    db=Depends(get_db),
):
    predictions = (
        db.query(Prediction)
        .all()
    )

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
    "/analytics",
    response_model=AnalyticsResponse,
)
def get_analytics(
    db=Depends(get_db),
):
    predictions = (
        db.query(Prediction)
        .order_by(
            Prediction.created_at.desc()
        )
        .all()
    )

    if not predictions:
        return {
            "total_predictions": 0,
            "average_predicted_bod5": 0.0,
            "minimum_predicted_bod5": 0.0,
            "maximum_predicted_bod5": 0.0,
            "status_counts": {
                "low": 0,
                "moderate": 0,
                "high": 0,
                "very_high": 0,
            },
            "recent_predictions": [],
        }

    values = [
        prediction.predicted_effluent_bod5
        for prediction in predictions
    ]

    status_counts = {
        "low": 0,
        "moderate": 0,
        "high": 0,
        "very_high": 0,
    }

    for value in values:
        status = classify_bod5(value)
        status_counts[status] += 1

    recent_predictions = [
        {
            "id": prediction.id,
            "influent_bod5": prediction.influent_bod5,
            "influent_cod": prediction.influent_cod,
            "influent_tss": prediction.influent_tss,
            "flow_m3_day": prediction.flow_m3_day,
            "dissolved_oxygen": prediction.dissolved_oxygen,
            "temperature": prediction.temperature,
            "hrt_hours": prediction.hrt_hours,
            "predicted_effluent_bod5": (
                prediction.predicted_effluent_bod5
            ),
            "status": classify_bod5(
                prediction.predicted_effluent_bod5
            ),
            "created_at": prediction.created_at,
        }
        for prediction in predictions[:10]
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
        "status_counts": status_counts,
        "recent_predictions": recent_predictions,
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
        .filter(
            Prediction.id == prediction_id
        )
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
        "predicted_effluent_bod5": (
            prediction.predicted_effluent_bod5
        ),
        "status": classify_bod5(
            prediction.predicted_effluent_bod5
        ),
        "created_at": prediction.created_at,
    }