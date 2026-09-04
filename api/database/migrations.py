from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine


RISK_ASSESSMENT_COLUMNS = {
    "decision_priority": "VARCHAR(20)",
    "decision_summary": "VARCHAR(1000)",
    "possible_contributors": "JSON",
    "checks_to_perform": "JSON",
    "recommended_actions": "JSON",
    "monitoring_recommendations": "JSON",
    "evidence": "JSON",
    "limitations": "JSON",
}


def migrate_risk_assessment_decision_fields(
    engine: Engine,
) -> list[str]:
    """
    Add V2.7 Decision Engine columns to the existing
    risk_assessments table when they do not already exist.

    Existing records are preserved.
    Newly added columns are nullable.
    """

    inspector = inspect(engine)

    if "risk_assessments" not in inspector.get_table_names():
        return []

    existing_columns = {
        column["name"]
        for column in inspector.get_columns("risk_assessments")
    }

    added_columns: list[str] = []

    with engine.begin() as connection:
        for column_name, column_type in RISK_ASSESSMENT_COLUMNS.items():
            if column_name in existing_columns:
                continue

            statement = text(
                f"""
                ALTER TABLE risk_assessments
                ADD COLUMN {column_name} {column_type}
                """
            )

            connection.execute(statement)
            added_columns.append(column_name)

    return added_columns
