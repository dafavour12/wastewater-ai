def test_get_api_url_default(monkeypatch):
    monkeypatch.delenv(
        "WASTEWATER_API_URL",
        raising=False,
    )

    from api.config import get_api_url

    assert (
        get_api_url()
        == "http://127.0.0.1:8000"
    )


def test_get_api_url_from_environment(monkeypatch):
    monkeypatch.setenv(
        "WASTEWATER_API_URL",
        "http://example.com/",
    )

    from api.config import get_api_url

    assert (
        get_api_url()
        == "http://example.com"
    )


def test_get_environment_default(monkeypatch):
    monkeypatch.delenv(
        "WASTEWATER_ENVIRONMENT",
        raising=False,
    )

    from api.config import get_environment

    assert (
        get_environment()
        == "development"
    )


def test_get_environment_from_environment(monkeypatch):
    monkeypatch.setenv(
        "WASTEWATER_ENVIRONMENT",
        "production",
    )

    from api.config import get_environment

    assert (
        get_environment()
        == "production"
    )


def test_get_database_url_default(monkeypatch):
    monkeypatch.delenv(
        "WASTEWATER_DATABASE_URL",
        raising=False,
    )

    from api.config import get_database_url

    assert (
        get_database_url()
        == "sqlite:///./wastewater.db"
    )


def test_get_database_url_from_environment(monkeypatch):
    monkeypatch.setenv(
        "WASTEWATER_DATABASE_URL",
        "sqlite:///./test.db",
    )

    from api.config import get_database_url

    assert (
        get_database_url()
        == "sqlite:///./test.db"
    )


def test_get_bod5_model_path_default(monkeypatch):
    monkeypatch.delenv(
        "WASTEWATER_BOD5_MODEL_PATH",
        raising=False,
    )

    from api.config import get_bod5_model_path

    assert (
        get_bod5_model_path()
        == "models/wastewater_bod5_model.joblib"
    )


def test_get_process_model_path_default(monkeypatch):
    monkeypatch.delenv(
        "WASTEWATER_PROCESS_MODEL_PATH",
        raising=False,
    )

    from api.config import get_process_model_path

    assert (
        get_process_model_path()
        == "models/v25_process_anomaly_model.joblib"
    )
