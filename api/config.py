from __future__ import annotations

import os


def get_api_url() -> str:
    """Return the configured API URL."""

    return os.getenv(
        "WASTEWATER_API_URL",
        "http://127.0.0.1:8000",
    ).rstrip("/")


def get_environment() -> str:
    """Return the current application environment."""

    return os.getenv(
        "WASTEWATER_ENVIRONMENT",
        "development",
    )


def get_database_url() -> str:
    """Return the configured database URL."""

    return os.getenv(
        "WASTEWATER_DATABASE_URL",
        "sqlite:///./wastewater.db",
    )


def get_bod5_model_path() -> str:
    """Return the configured BOD5 model path."""

    return os.getenv(
        "WASTEWATER_BOD5_MODEL_PATH",
        "models/wastewater_bod5_model.joblib",
    )


def get_process_model_path() -> str:
    """Return the configured process anomaly model path."""

    return os.getenv(
        "WASTEWATER_PROCESS_MODEL_PATH",
        "models/v25_process_anomaly_model.joblib",
    )
