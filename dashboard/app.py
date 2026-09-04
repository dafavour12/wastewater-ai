import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"


PROCESS_FIELDS = [
    ("PH-P", "pH - Primary"),
    ("DBO-P", "BOD5 - Primary (mg/L)"),
    ("SS-P", "Suspended Solids - Primary (mg/L)"),
    ("SSV-P", "Volatile Suspended Solids - Primary (mg/L)"),
    ("SED-P", "Sedimentation - Primary"),
    ("COND-P", "Conductivity - Primary"),
    ("PH-D", "pH - Digester"),
    ("DBO-D", "BOD5 - Digester (mg/L)"),
    ("DQO-D", "COD - Digester (mg/L)"),
    ("SS-D", "Suspended Solids - Digester (mg/L)"),
    ("SSV-D", "Volatile Suspended Solids - Digester (mg/L)"),
    ("SED-D", "Sedimentation - Digester"),
    ("COND-D", "Conductivity - Digester"),
    ("RD-DBO-P", "BOD5 Reduction - Primary"),
    ("RD-SS-P", "SS Reduction - Primary"),
    ("RD-DBO-D", "BOD5 Reduction - Digester"),
    ("RD-SS-D", "SS Reduction - Digester"),
    ("RD-DBO-G", "BOD5 Reduction - Global"),
    ("RD-SS-G", "SS Reduction - Global"),
    ("RD-SED-G", "Sedimentation Reduction - Global"),
    ("RD-N-NH4", "Ammonium Reduction"),
    ("RD-N-NO2", "Nitrite Reduction"),
]


def api_get(path: str):
    """Send a GET request to the FastAPI backend."""

    response = requests.get(
        f"{API_URL}{path}",
        timeout=10,
    )

    response.raise_for_status()

    return response.json()


def api_post(path: str, payload: dict):
    """Send a POST request to the FastAPI backend."""

    response = requests.post(
        f"{API_URL}{path}",
        json=payload,
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


def get_risk_statistics():
    """Get persistent risk-assessment statistics."""

    return api_get("/risk/assessments/stats")


def get_risk_history():
    """Get persistent risk-assessment history."""

    return api_get("/risk/assessments")


def get_risk_assessment(assessment_id: int):
    """Get one persisted risk assessment."""

    return api_get(
        f"/risk/assessments/{assessment_id}"
    )


def submit_risk_assessment(payload: dict):
    """Submit a complete wastewater/process risk assessment."""

    return api_post(
        "/risk/assess/process",
        payload,
    )


def risk_level_label(level: str) -> str:
    """Return a readable risk-level label."""

    return str(level).replace("_", " ").upper()


def risk_level_indicator(level: str) -> str:
    """Return a simple visual indicator for the risk level."""

    indicators = {
        "NORMAL": "🟢",
        "LOW": "🔵",
        "ELEVATED": "🟡",
        "HIGH": "🟠",
        "CRITICAL": "🔴",
    }

    return indicators.get(
        risk_level_label(level),
        "⚪",
    )


def decision_priority_indicator(priority: str) -> str:
    """Return a visual indicator for decision priority."""

    indicators = {
        "NORMAL": "🟢",
        "LOW": "🔵",
        "MODERATE": "🟡",
        "HIGH": "🟠",
        "CRITICAL": "🔴",
    }

    return indicators.get(
        risk_level_label(priority),
        "⚪",
    )


def format_number(value, decimals: int = 2) -> str:
    """Format numeric dashboard values safely."""

    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def render_metric_row(
    total_assessments,
    average_risk,
    maximum_risk,
    anomaly_count,
):
    """Render dashboard summary metrics."""

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Assessments",
            total_assessments,
        )

    with col2:
        st.metric(
            "Average Risk Score",
            format_number(average_risk),
        )

    with col3:
        st.metric(
            "Maximum Risk Score",
            format_number(maximum_risk),
        )

    with col4:
        st.metric(
            "Anomalies Detected",
            anomaly_count,
        )


def render_risk_distribution(risk_level_counts: dict):
    """Render risk-level distribution without dataframe/chart dependencies."""

    st.subheader("Risk Distribution")

    levels = [
        "NORMAL",
        "LOW",
        "ELEVATED",
        "HIGH",
        "CRITICAL",
    ]

    total = sum(
        int(risk_level_counts.get(level, 0))
        for level in levels
    )

    if total == 0:
        st.info("No risk assessments have been recorded yet.")
        return

    for level in levels:
        count = int(
            risk_level_counts.get(
                level,
                0,
            )
        )

        percentage = (
            count / total * 100.0
            if total
            else 0.0
        )

        st.write(
            f"{risk_level_indicator(level)} "
            f"**{level}** — {count} "
            f"({percentage:.1f}%)"
        )

        st.progress(
            min(
                count / total,
                1.0,
            )
        )


def render_assessment_table(assessments: list):
    """Render recent assessments as a lightweight Markdown table."""

    st.subheader("Recent Risk Assessments")

    if not assessments:
        st.info("No risk assessments have been recorded yet.")
        return

    recent = assessments[:20]

    header = (
        "| ID | Created | Predicted BOD₅ | "
        "Anomaly | Risk | Score | Decision |\n"
        "|---:|---|---:|---|---|---:|---|"
    )

    rows = []

    for assessment in recent:
        assessment_id = assessment.get(
            "id",
            "N/A",
        )

        created_at = assessment.get(
            "created_at",
            "N/A",
        )

        if isinstance(created_at, str):
            created_at = created_at.replace(
                "T",
                " ",
            )

        predicted_bod5 = format_number(
            assessment.get(
                "predicted_effluent_bod5"
            )
        )

        anomaly = (
            "Yes"
            if assessment.get(
                "is_anomaly",
                False,
            )
            else "No"
        )

        risk = risk_level_label(
            assessment.get(
                "overall_risk_level",
                "UNKNOWN",
            )
        )

        score = format_number(
            assessment.get(
                "overall_risk_score"
            )
        )

        decision_priority = risk_level_label(
            assessment.get(
                "decision_priority",
                "N/A",
            )
        )

        rows.append(
            f"| {assessment_id} | "
            f"{created_at} | "
            f"{predicted_bod5} | "
            f"{anomaly} | "
            f"{risk} | "
            f"{score} | "
            f"{decision_priority} |"
        )

    st.markdown(
        header
        + "\n"
        + "\n".join(rows)
    )


def render_list_section(
    title: str,
    items,
    empty_message: str,
):
    """Render a list returned by the decision engine."""

    st.markdown(f"#### {title}")

    if not items:
        st.write(empty_message)
        return

    for item in items:
        st.markdown(
            f"- {item}"
        )


def render_decision_details(assessment: dict):
    """Render V2.7 Decision Engine output."""

    decision_priority = risk_level_label(
        assessment.get(
            "decision_priority",
            "N/A",
        )
    )

    decision_summary = assessment.get(
        "decision_summary",
        "No decision summary available.",
    )

    st.subheader("Decision & Recommendations")

    st.markdown(
        f"### "
        f"{decision_priority_indicator(decision_priority)} "
        f"{decision_priority} PRIORITY"
    )

    st.info(
        decision_summary
    )

    col1, col2 = st.columns(2)

    with col1:
        render_list_section(
            "Possible Contributors",
            assessment.get(
                "possible_contributors",
                [],
            ),
            "No possible contributors were identified.",
        )

    with col2:
        render_list_section(
            "Checks to Perform",
            assessment.get(
                "checks_to_perform",
                [],
            ),
            "No additional checks were specified.",
        )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        render_list_section(
            "Recommended Actions",
            assessment.get(
                "recommended_actions",
                [],
            ),
            "No recommended actions were specified.",
        )

    with col2:
        render_list_section(
            "Monitoring Recommendations",
            assessment.get(
                "monitoring_recommendations",
                [],
            ),
            "No monitoring recommendations were specified.",
        )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        render_list_section(
            "Evidence",
            assessment.get(
                "evidence",
                [],
            ),
            "No evidence details were recorded.",
        )

    with col2:
        render_list_section(
            "Limitations",
            assessment.get(
                "limitations",
                [],
            ),
            "No limitations were recorded.",
        )


def render_assessment_details(assessment: dict):
    """Render detailed information for one assessment."""

    if not assessment:
        return

    level = risk_level_label(
        assessment.get(
            "overall_risk_level",
            "UNKNOWN",
        )
    )

    st.subheader("Assessment Details")

    st.markdown(
        f"### {risk_level_indicator(level)} "
        f"{level} RISK"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Risk Score",
            format_number(
                assessment.get(
                    "overall_risk_score"
                )
            ),
        )

    with col2:
        st.metric(
            "Predicted Effluent BOD₅",
            f"{format_number(assessment.get('predicted_effluent_bod5'))} mg/L",
        )

    with col3:
        st.metric(
            "Anomaly Percentile",
            f"{format_number(assessment.get('anomaly_percentile'))}%",
        )

    st.write(
        f"**Prediction status:** "
        f"{assessment.get('prediction_status', 'N/A')}"
    )

    st.write(
        f"**Anomaly status:** "
        f"{'Detected' if assessment.get('is_anomaly') else 'Not detected'}"
    )

    st.write(
        f"**Anomaly risk band:** "
        f"{risk_level_label(assessment.get('anomaly_risk_band', 'N/A'))}"
    )

    st.write(
        f"**Alert level:** "
        f"{risk_level_label(assessment.get('anomaly_alert_level', 'N/A'))}"
    )

    st.write(
        f"**Prediction score:** "
        f"{format_number(assessment.get('prediction_score'))}"
    )

    st.write(
        f"**Confidence score:** "
        f"{format_number(assessment.get('confidence_score'))}"
    )

    st.write(
        f"**Model confidence:** "
        f"{assessment.get('model_confidence', 'N/A')}"
    )

    st.write(
        f"**Monitoring method:** "
        f"{assessment.get('monitoring_method', 'N/A')}"
    )

    st.write(
        f"**Contamination setting:** "
        f"{format_number(assessment.get('contamination'), 3)}"
    )

    st.write(
        f"**Process features used:** "
        f"{assessment.get('process_features_used', 'N/A')}"
    )

    st.markdown("#### Risk Reason")

    st.info(
        assessment.get(
            "risk_reason",
            "No risk reason available.",
        )
    )

    st.markdown("#### Recommended Action")

    st.warning(
        assessment.get(
            "recommended_action",
            "No recommended action available.",
        )
    )

    st.divider()

    render_decision_details(
        assessment
    )


def render_dashboard():
    """Render the complete Wastewater AI monitoring dashboard."""

    st.set_page_config(
        page_title="Wastewater AI",
        page_icon="💧",
        layout="wide",
    )

    st.title("💧 Wastewater AI")
    st.caption(
        "AI-assisted wastewater treatment prediction, "
        "process monitoring, and risk assessment."
    )

    st.sidebar.header("Navigation")

    page = st.sidebar.radio(
        "Select view",
        [
            "Risk Monitoring",
            "New Risk Assessment",
        ],
    )

    if page == "Risk Monitoring":
        render_monitoring_page()
    else:
        render_new_assessment_page()


def render_monitoring_page():
    """Render the persistent risk-monitoring page."""

    st.header("Risk Monitoring")

    try:
        statistics = get_risk_statistics()
        history = get_risk_history()

    except requests.RequestException as exc:
        st.error(
            "Unable to connect to the FastAPI backend."
        )

        st.code(
            str(exc)
        )

        st.info(
            "Start the API with: "
            "`python -m uvicorn api.main:app --reload`"
        )

        return

    render_metric_row(
        statistics.get(
            "total_assessments",
            0,
        ),
        statistics.get(
            "average_risk_score",
            0,
        ),
        statistics.get(
            "maximum_risk_score",
            0,
        ),
        statistics.get(
            "anomaly_count",
            0,
        ),
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        render_risk_distribution(
            statistics.get(
                "risk_level_counts",
                {},
            )
        )

    with col2:
        st.subheader("Latest Assessment")

        if history:
            latest_id = history[0].get("id")

            try:
                latest = get_risk_assessment(
                    latest_id
                )

                level = risk_level_label(
                    latest.get(
                        "overall_risk_level",
                        "UNKNOWN",
                    )
                )

                decision_priority = risk_level_label(
                    latest.get(
                        "decision_priority",
                        "N/A",
                    )
                )

                st.markdown(
                    f"## "
                    f"{risk_level_indicator(level)} "
                    f"{level}"
                )

                st.metric(
                    "Risk Score",
                    format_number(
                        latest.get(
                            "overall_risk_score"
                        )
                    ),
                )

                st.metric(
                    "Predicted Effluent BOD₅",
                    (
                        f"{format_number(latest.get('predicted_effluent_bod5'))} "
                        "mg/L"
                    ),
                )

                st.markdown(
                    f"**Decision Priority:** "
                    f"{decision_priority_indicator(decision_priority)} "
                    f"{decision_priority}"
                )

                st.markdown("**Decision Summary**")

                st.info(
                    latest.get(
                        "decision_summary",
                        "No decision summary available.",
                    )
                )

                st.markdown("**Recommended Actions**")

                recommended_actions = latest.get(
                    "recommended_actions",
                    [],
                )

                if recommended_actions:
                    for action in recommended_actions:
                        st.markdown(
                            f"- {action}"
                        )
                else:
                    st.write(
                        "No recommended actions available."
                    )

            except requests.RequestException:
                st.warning(
                    "Latest assessment details could not be loaded."
                )
        else:
            st.info(
                "No assessments have been recorded yet."
            )

    st.divider()

    render_assessment_table(history)

    if history:
        st.subheader("View Assessment")

        assessment_ids = [
            assessment.get("id")
            for assessment in history
            if assessment.get("id") is not None
        ]

        if assessment_ids:
            selected_id = st.selectbox(
                "Select assessment ID",
                assessment_ids,
            )

            if st.button(
                "Load Assessment Details"
            ):
                try:
                    assessment = get_risk_assessment(
                        selected_id
                    )

                    render_assessment_details(
                        assessment
                    )

                except requests.RequestException as exc:
                    st.error(
                        "Unable to load the selected assessment."
                    )

                    st.code(
                        str(exc)
                    )


def render_new_assessment_page():
    """Render the integrated risk-assessment input page."""

    st.header("New Risk Assessment")

    st.write(
        "Enter wastewater and process measurements. "
        "Wastewater AI will predict effluent BOD₅, "
        "detect process anomalies, calculate the combined "
        "risk, generate an engineering decision, and persist "
        "the assessment."
    )

    st.subheader("Wastewater Conditions")

    col1, col2 = st.columns(2)

    with col1:
        influent_bod5 = st.number_input(
            "Influent BOD₅ (mg/L)",
            min_value=0.0,
            value=300.0,
        )

        influent_cod = st.number_input(
            "Influent COD (mg/L)",
            min_value=0.0,
            value=560.0,
        )

        influent_tss = st.number_input(
            "Influent TSS (mg/L)",
            min_value=0.0,
            value=250.0,
        )

        flow_m3_day = st.number_input(
            "Flow (m³/day)",
            min_value=0.0,
            value=1050.0,
        )

    with col2:
        dissolved_oxygen = st.number_input(
            "Dissolved Oxygen (mg/L)",
            min_value=0.0,
            value=2.1,
        )

        temperature = st.number_input(
            "Temperature (°C)",
            min_value=0.0,
            value=27.0,
        )

        hrt_hours = st.number_input(
            "HRT (hours)",
            min_value=0.0,
            value=8.0,
        )

        model_confidence = st.selectbox(
            "Model Confidence",
            [
                "research",
                "high",
                "medium",
                "low",
            ],
            index=0,
        )

    st.subheader("Process Monitoring Variables")

    process = {}

    columns = st.columns(2)

    for index, (api_name, label) in enumerate(
        PROCESS_FIELDS
    ):
        with columns[index % 2]:
            process[api_name] = st.number_input(
                label,
                min_value=0.0,
                value=0.0,
                key=f"process_{api_name}",
            )

    st.divider()

    if st.button(
        "Run Risk Assessment",
        type="primary",
    ):
        payload = {
            "wastewater": {
                "influent_bod5": influent_bod5,
                "influent_cod": influent_cod,
                "influent_tss": influent_tss,
                "flow_m3_day": flow_m3_day,
                "dissolved_oxygen": dissolved_oxygen,
                "temperature": temperature,
                "hrt_hours": hrt_hours,
            },
            "process": process,
            "model_confidence": model_confidence,
        }

        try:
            with st.spinner(
                "Running wastewater prediction, "
                "process monitoring, risk assessment, "
                "and decision engine..."
            ):
                result = submit_risk_assessment(
                    payload
                )

            st.success(
                "Risk assessment completed, decision generated, "
                "and assessment persisted."
            )

            assessment_id = result.get(
                "assessment_id"
            )

            if assessment_id is not None:
                st.write(
                    f"**Assessment ID:** {assessment_id}"
                )

            render_assessment_details(
                result
            )

        except requests.HTTPError as exc:
            st.error(
                "The API rejected the risk assessment."
            )

            if exc.response is not None:
                try:
                    st.json(
                        exc.response.json()
                    )
                except ValueError:
                    st.code(
                        exc.response.text
                    )

        except requests.RequestException as exc:
            st.error(
                "Unable to connect to the FastAPI backend."
            )

            st.code(
                str(exc)
            )


if __name__ == "__main__":
    render_dashboard()
