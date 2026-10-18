from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "FINAL_MODEL.joblib"
RESULTS_PATH = BASE_DIR / "FINAL_MODEL_results.csv"
REPORT_PATH = BASE_DIR / "FINAL_MODEL_classification_report.csv"

DIAGNOSTIC_IMAGES = {
    "ROC Curve": BASE_DIR / "FINAL_MODEL_roc_curve.png",
    "Precision-Recall Curve": BASE_DIR / "FINAL_MODEL_precision_recall.png",
    "Threshold Trade-offs": BASE_DIR / "FINAL_MODEL_threshold_analysis.png",
    "Calibration Curve": BASE_DIR / "FINAL_MODEL_calibration_curve.png",
    "Probability Distribution": BASE_DIR / "FINAL_MODEL_probability_distribution.png",
    "Learning Curve": BASE_DIR / "FINAL_MODEL_learning_curve.png",
    "Feature Importance": BASE_DIR / "FINAL_MODEL_feature_importance.png",
    "Confusion Matrix": BASE_DIR / "FINAL_MODEL_confusion_matrix.png",
}

EXPECTED_FALLBACK_FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

FEATURE_SPECS: dict[str, dict[str, Any]] = {
    "age": {
        "label": "Age (years)",
        "represents": "Patient age at evaluation time",
        "unit": "years",
        "kind": "int",
        "min": 20,
        "max": 90,
        "default": 54,
        "help": "Patient age in years.",
    },
    "sex": {
        "label": "Sex",
        "represents": "Biological sex category",
        "unit": "category code (0=female, 1=male)",
        "kind": "select",
        "options": [0, 1],
        "default": 1,
        "help": "Biological sex.",
    },
    "cp": {
        "label": "Chest Pain Type",
        "represents": "Type of chest pain symptom",
        "unit": "category code (1-4)",
        "kind": "select",
        "options": [1, 2, 3, 4],
        "default": 3,
        "help": "1=Typical angina, 2=Atypical, 3=Non-anginal, 4=Asymptomatic.",
    },
    "trestbps": {
        "label": "Resting Blood Pressure (mm Hg)",
        "represents": "Resting systolic blood pressure",
        "unit": "mm Hg",
        "kind": "int",
        "min": 80,
        "max": 220,
        "default": 130,
        "help": "Resting blood pressure measured on admission.",
    },
    "chol": {
        "label": "Serum Cholesterol (mg/dL)",
        "represents": "Serum cholesterol concentration",
        "unit": "mg/dL",
        "kind": "int",
        "min": 100,
        "max": 600,
        "default": 246,
        "help": "Serum cholesterol in mg/dL.",
    },
    "fbs": {
        "label": "Fasting Blood Sugar > 120 mg/dL",
        "represents": "Whether fasting blood sugar exceeds 120 mg/dL",
        "unit": "binary flag (0=no, 1=yes)",
        "kind": "select",
        "options": [0, 1],
        "default": 0,
        "help": "1=True, 0=False.",
    },
    "restecg": {
        "label": "Resting ECG",
        "represents": "Resting electrocardiogram category",
        "unit": "category code (0-2)",
        "kind": "select",
        "options": [0, 1, 2],
        "default": 1,
        "help": "Resting electrocardiographic result category.",
    },
    "thalach": {
        "label": "Max Heart Rate Achieved",
        "represents": "Maximum heart rate during exercise",
        "unit": "beats per minute (bpm)",
        "kind": "int",
        "min": 60,
        "max": 220,
        "default": 150,
        "help": "Maximum heart rate during exercise.",
    },
    "exang": {
        "label": "Exercise-Induced Angina",
        "represents": "Presence of exercise-induced angina",
        "unit": "binary flag (0=no, 1=yes)",
        "kind": "select",
        "options": [0, 1],
        "default": 0,
        "help": "1=Yes, 0=No.",
    },
    "oldpeak": {
        "label": "ST Depression (oldpeak)",
        "represents": "ST depression induced by exercise relative to rest",
        "unit": "mm ST depression",
        "kind": "float",
        "min": 0.0,
        "max": 6.5,
        "default": 1.0,
        "step": 0.1,
        "help": "ST depression induced by exercise relative to rest.",
    },
    "slope": {
        "label": "Slope of ST Segment",
        "represents": "Slope shape of peak exercise ST segment",
        "unit": "category code (1-3)",
        "kind": "select",
        "options": [1, 2, 3],
        "default": 2,
        "help": "1=Upsloping, 2=Flat, 3=Downsloping.",
    },
    "ca": {
        "label": "Number of Major Vessels (0-4)",
        "represents": "Number of major vessels observed by fluoroscopy",
        "unit": "count",
        "kind": "select",
        "options": [0, 1, 2, 3, 4],
        "default": 0,
        "help": "Count of major vessels colored by fluoroscopy.",
    },
    "thal": {
        "label": "Thalassemia",
        "represents": "Thalassemia status from thallium stress test",
        "unit": "category code (3, 6, 7)",
        "kind": "select",
        "options": [3, 6, 7],
        "default": 3,
        "help": "3=Normal, 6=Fixed defect, 7=Reversible defect.",
    },
}

CATEGORY_LABELS: dict[str, dict[int, str]] = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {
        1: "Typical angina",
        2: "Atypical angina",
        3: "Non-anginal pain",
        4: "Asymptomatic",
    },
    "fbs": {0: "No", 1: "Yes"},
    "restecg": {
        0: "Normal",
        1: "ST-T wave abnormality",
        2: "Left ventricular hypertrophy",
    },
    "exang": {0: "No", 1: "Yes"},
    "slope": {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
    "thal": {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"},
}


@st.cache_resource(show_spinner=False)
def load_model(model_path: Path):
    return joblib.load(model_path)


@st.cache_data(show_spinner=False)
def load_results_dict(results_path: Path) -> dict[str, Any]:
    if not results_path.exists():
        return {}
    results_df = pd.read_csv(results_path)
    if results_df.empty:
        return {}
    return results_df.iloc[0].to_dict()


@st.cache_data(show_spinner=False)
def load_classification_report(report_path: Path) -> pd.DataFrame:
    if not report_path.exists():
        return pd.DataFrame()
    return pd.read_csv(report_path)


def apply_custom_style() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Manrope:wght@400;500;700&display=swap');

        :root {
            --brand-teal: #0b7a75;
            --brand-ink: #1b2636;
            --brand-gold: #d08c00;
            --brand-coral: #bb3e3e;
            --surface: rgba(255, 255, 255, 0.82);
            --surface-strong: rgba(255, 255, 255, 0.92);
            --border-soft: rgba(27, 38, 54, 0.12);
        }

        .stApp {
            background:
                radial-gradient(circle at 8% 10%, rgba(11, 122, 117, 0.22), transparent 35%),
                radial-gradient(circle at 88% 22%, rgba(208, 140, 0, 0.25), transparent 33%),
                linear-gradient(140deg, #fffaf0 0%, #f3fbff 52%, #f7f2e8 100%);
            color: var(--brand-ink);
            font-family: 'Manrope', sans-serif;
        }

        .stApp p,
        .stApp li,
        .stApp label,
        .stApp h4,
        .stApp h5,
        .stApp h6,
        .stApp [data-testid="stMarkdownContainer"] {
            color: #1b2636;
        }

        .block-container {
            max-width: 1220px;
            padding-top: 1.2rem;
            padding-bottom: 2.2rem;
        }

        h1, h2, h3 {
            font-family: 'Space Grotesk', sans-serif;
            letter-spacing: -0.02em;
            color: #152233;
        }

        .hero-panel {
            background: var(--surface);
            border: 1px solid var(--border-soft);
            border-radius: 22px;
            padding: 1.3rem 1.5rem;
            box-shadow: 0 18px 48px rgba(15, 25, 35, 0.10);
            animation: fadeUp 650ms ease-out;
        }

        .hero-kicker {
            color: var(--brand-teal);
            font-size: 0.86rem;
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 0.09em;
            margin-bottom: 0.35rem;
        }

        .hero-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: clamp(1.75rem, 2.2vw, 2.35rem);
            font-weight: 700;
            margin: 0;
            color: #152233;
        }

        .hero-subtitle {
            margin-top: 0.55rem;
            font-size: 1rem;
            line-height: 1.45;
            color: #2a384a;
        }

        .badge-row {
            display: flex;
            gap: 0.55rem;
            flex-wrap: wrap;
            margin-top: 0.9rem;
        }

        .hero-badge {
            background: rgba(11, 122, 117, 0.10);
            border: 1px solid rgba(11, 122, 117, 0.28);
            border-radius: 999px;
            padding: 0.32rem 0.72rem;
            font-size: 0.82rem;
            color: #114542;
            font-weight: 600;
        }

        .result-card {
            border-radius: 18px;
            padding: 1rem 1.1rem;
            border: 1px solid rgba(255, 255, 255, 0.45);
            box-shadow: 0 8px 24px rgba(15, 25, 35, 0.08);
            margin-bottom: 0.8rem;
            animation: fadeUp 650ms ease-out;
        }

        .result-card.safe {
            background: linear-gradient(130deg, rgba(61, 153, 112, 0.17), rgba(255, 255, 255, 0.88));
            border-color: rgba(61, 153, 112, 0.32);
        }

        .result-card.risk {
            background: linear-gradient(130deg, rgba(187, 62, 62, 0.16), rgba(255, 255, 255, 0.90));
            border-color: rgba(187, 62, 62, 0.30);
        }

        .result-title {
            margin: 0;
            font-weight: 700;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.12rem;
        }

        .result-text {
            margin: 0.35rem 0 0;
            color: #2b3848;
            line-height: 1.45;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(170deg, rgba(255, 255, 255, 0.92), rgba(238, 246, 252, 0.95));
            border-right: 1px solid var(--border-soft);
        }

        [data-testid="stSidebar"] * {
            color: #1b2636 !important;
        }

        [data-testid="stMetric"] {
            background: var(--surface-strong);
            border: 1px solid var(--border-soft);
            border-radius: 14px;
            padding: 0.6rem 0.85rem;
            box-shadow: 0 10px 20px rgba(15, 25, 35, 0.05);
            min-height: 90px;
        }

        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"] {
            color: #1b2636 !important;
        }

        .stSlider label,
        .stNumberInput label,
        .stSelectbox label,
        .stFileUploader label,
        .stTextInput label {
            color: #14253a !important;
            font-weight: 700 !important;
        }

        .stSelectbox [data-baseweb="select"] > div,
        .stNumberInput input,
        .stTextInput input {
            background: rgba(255, 255, 255, 0.95) !important;
            color: #1b2636 !important;
            border: 1px solid rgba(27, 38, 54, 0.24) !important;
            border-radius: 10px !important;
        }

        .stSelectbox svg {
            fill: #1b2636 !important;
        }

        .stSlider [data-baseweb="slider"] [role="slider"] {
            background: #0b7a75 !important;
            border-color: #0b7a75 !important;
        }

        .stSlider [data-baseweb="slider"] > div > div {
            background: #0b7a75 !important;
        }

        .stSlider [data-testid="stTickBarMin"],
        .stSlider [data-testid="stTickBarMax"],
        .stSlider [data-testid="stTickBar"] {
            color: #31465e !important;
        }

        [data-testid="stFileUploaderDropzone"] {
            background: rgba(255, 255, 255, 0.86);
            border: 1px dashed rgba(27, 38, 54, 0.30);
        }

        [data-testid="stFileUploaderDropzone"] * {
            color: #1b2636 !important;
        }

        .stAlert {
            color: #1b2636;
        }

        .feature-meta {
            margin: 0.28rem 0 0.85rem;
            padding: 0.42rem 0.52rem;
            font-size: 0.77rem;
            line-height: 1.4;
            border-radius: 10px;
            border: 1px solid rgba(11, 122, 117, 0.24);
            background: rgba(11, 122, 117, 0.08);
            color: #1d344d;
        }

        .feature-meta strong {
            color: #10354c;
            font-weight: 700;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
            margin-bottom: 0.65rem;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.65);
            border: 1px solid var(--border-soft);
            border-radius: 12px;
            padding: 0.42rem 0.8rem;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 600;
            color: #1f3044;
        }

        .stTabs [aria-selected="true"] {
            background: #0b7a75 !important;
            color: #ffffff !important;
            border-color: #0b7a75 !important;
        }

        .stDownloadButton button,
        .stButton button,
        .stFormSubmitButton button {
            border-radius: 12px;
            border: 1px solid rgba(11, 122, 117, 0.45);
            background: linear-gradient(120deg, #0b7a75, #0f8f88);
            color: #ffffff;
            font-weight: 700;
            font-family: 'Manrope', sans-serif;
        }

        .stDownloadButton button:hover,
        .stButton button:hover,
        .stFormSubmitButton button:hover {
            border-color: rgba(11, 122, 117, 0.75);
            background: linear-gradient(120deg, #0d837d, #13a199);
            color: #ffffff;
        }

        @keyframes fadeUp {
            0% {
                opacity: 0;
                transform: translateY(12px);
            }
            100% {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @media (max-width: 900px) {
            .hero-panel {
                padding: 1rem 1rem;
            }
            .hero-subtitle {
                font-size: 0.95rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_percent(value: Any) -> str:
    val = as_float(value, np.nan)
    if np.isnan(val):
        return "N/A"
    return f"{val:.2%}"


def as_metric(value: Any, digits: int = 4) -> str:
    val = as_float(value, np.nan)
    if np.isnan(val):
        return "N/A"
    return f"{val:.{digits}f}"


def get_expected_features(model: Any) -> list[str]:
    if hasattr(model, "feature_names_in_"):
        return [str(name) for name in model.feature_names_in_]
    return EXPECTED_FALLBACK_FEATURES.copy()


def normalize_input_frame(frame: pd.DataFrame, expected_features: list[str]) -> pd.DataFrame:
    normalized = frame.copy()
    for col in expected_features:
        if col not in normalized.columns:
            normalized[col] = np.nan
    normalized = normalized[expected_features]
    for col in normalized.columns:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
    return normalized


def extract_feature_importance(model: Any, top_n: int = 18) -> pd.DataFrame:
    if not hasattr(model, "named_steps"):
        return pd.DataFrame()

    try:
        rf_model = model.named_steps["model"]
        preprocessor = model.named_steps["preprocess"]
    except Exception:
        return pd.DataFrame()

    if not hasattr(rf_model, "feature_importances_"):
        return pd.DataFrame()

    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = [f"feature_{idx}" for idx in range(len(rf_model.feature_importances_))]

    if len(feature_names) != len(rf_model.feature_importances_):
        min_len = min(len(feature_names), len(rf_model.feature_importances_))
        feature_names = feature_names[:min_len]
        importances = rf_model.feature_importances_[:min_len]
    else:
        importances = rf_model.feature_importances_

    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return importance_df.sort_values("importance", ascending=False).head(top_n)


def render_hero(results: dict[str, Any], threshold: float) -> None:
    accuracy_text = as_percent(results.get("accuracy"))
    roc_auc_text = as_percent(results.get("roc_auc"))
    f1_text = as_percent(results.get("f1"))

    st.markdown(
        f"""
        <section class="hero-panel">
            <div class="hero-kicker">Cardiovascular Risk Decision Support</div>
            <h1 class="hero-title">CardioScope Clinical AI Console</h1>
            <p class="hero-subtitle">
                Production interface for your trained Random Forest model. Run single-patient inference,
                perform batch scoring, and inspect model quality diagnostics in one streamlined app.
            </p>
            <div class="badge-row">
                <span class="hero-badge">Accuracy: {accuracy_text}</span>
                <span class="hero-badge">ROC-AUC: {roc_auc_text}</span>
                <span class="hero-badge">F1 Score: {f1_text}</span>
                <span class="hero-badge">Decision Threshold: {threshold:.2f}</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_feature_meta(spec: dict[str, Any]) -> None:
    representation = spec.get("represents", "Clinical measurement")
    unit = spec.get("unit", "not specified")
    st.markdown(
        (
            "<div class='feature-meta'>"
            f"<strong>Represents:</strong> {representation}<br>"
            f"<strong>Unit:</strong> {unit}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_input_widget(feature: str) -> Any:
    spec = FEATURE_SPECS.get(feature)
    if spec is None:
        return st.number_input(feature, value=0.0, step=1.0, key=f"input_{feature}")

    label = spec["label"]
    help_text = spec.get("help")
    kind = spec["kind"]
    unit = spec.get("unit", "")

    if kind in {"int", "float"} and unit and unit.lower() not in label.lower():
        label = f"{label} ({unit})"

    full_help = help_text
    if unit:
        full_help = f"{help_text} Unit: {unit}" if help_text else f"Unit: {unit}"

    if kind == "select":
        options = spec["options"]
        default_value = spec["default"]
        labels = CATEGORY_LABELS.get(feature, {})
        default_idx = options.index(default_value) if default_value in options else 0
        selected = st.selectbox(
            label,
            options,
            index=default_idx,
            format_func=lambda value: f"{value} - {labels.get(value, 'Category')}" if labels else str(value),
            help=full_help,
            key=f"input_{feature}",
        )
        render_feature_meta(spec)
        return selected

    if kind == "float":
        value = float(
            st.slider(
                label,
                min_value=float(spec["min"]),
                max_value=float(spec["max"]),
                value=float(spec["default"]),
                step=float(spec.get("step", 0.1)),
                format="%.1f",
                help=full_help,
                key=f"input_{feature}",
            )
        )
        render_feature_meta(spec)
        return value

    value = int(
        st.slider(
            label,
            min_value=int(spec["min"]),
            max_value=int(spec["max"]),
            value=int(spec["default"]),
            step=int(spec.get("step", 1)),
            format="%d",
            help=full_help,
            key=f"input_{feature}",
        )
    )
    render_feature_meta(spec)
    return value


def render_probability_chart(probability: float, threshold: float) -> None:
    if go is None:
        st.progress(max(0.0, min(1.0, probability)))
        st.caption(f"Risk Probability: {probability:.1%} | Threshold: {threshold:.1%}")
        return

    figure = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 34}},
            delta={"reference": threshold * 100, "increasing": {"color": "#bb3e3e"}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#0b7a75"},
                "steps": [
                    {"range": [0, threshold * 100], "color": "#d8efe8"},
                    {"range": [threshold * 100, 100], "color": "#fde1cf"},
                ],
                "threshold": {
                    "line": {"color": "#bb3e3e", "width": 4},
                    "thickness": 0.8,
                    "value": threshold * 100,
                },
            },
        )
    )
    figure.update_layout(
        height=290,
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Manrope, sans-serif", "color": "#1b2636"},
    )
    st.plotly_chart(figure, use_container_width=True)


def clinical_guidance(values: dict[str, Any], probability: float) -> list[str]:
    notes: list[str] = []

    if values.get("trestbps", 0) >= 140:
        notes.append("Resting blood pressure is elevated. Consider BP control planning and follow-up.")
    if values.get("chol", 0) >= 240:
        notes.append("Cholesterol is high. Lipid profile review and lifestyle intervention may be warranted.")
    if values.get("oldpeak", 0.0) >= 2.0:
        notes.append("ST depression is elevated. This signal often warrants additional cardiac assessment.")
    if values.get("exang", 0) == 1:
        notes.append("Exercise-induced angina reported. This is a meaningful risk feature in this model.")

    if probability >= 0.75:
        notes.append("Predicted risk is high. Prioritize confirmatory testing and specialist consultation.")
    elif probability >= 0.55:
        notes.append("Predicted risk is moderate. Use clinical context and consider additional diagnostics.")
    else:
        notes.append("Predicted risk is lower, but clinical judgement and patient history remain essential.")

    notes.append("This tool supports decision-making and is not a standalone medical diagnosis.")
    return notes


def render_single_prediction_tab(model: Any, expected_features: list[str], threshold: float) -> None:
    st.subheader("Single-Patient Inference")
    st.write("Enter patient measurements and run a real-time risk assessment.")

    with st.form("single_prediction_form", clear_on_submit=False):
        input_values: dict[str, Any] = {}
        columns = st.columns(3)

        for idx, feature in enumerate(expected_features):
            with columns[idx % 3]:
                input_values[feature] = render_input_widget(feature)

        submitted = st.form_submit_button("Run Risk Prediction", use_container_width=True)

    if not submitted:
        return

    input_frame = pd.DataFrame([input_values])
    input_frame = normalize_input_frame(input_frame, expected_features)

    probability = float(model.predict_proba(input_frame)[0, 1])
    prediction = int(probability >= threshold)

    result_class = "risk" if prediction == 1 else "safe"
    result_title = "Elevated Heart Disease Risk" if prediction == 1 else "Lower Heart Disease Risk"
    result_desc = (
        "Model probability is above the optimized threshold. Clinical follow-up is recommended."
        if prediction == 1
        else "Model probability is below the threshold. Continue with routine clinical context checks."
    )

    st.markdown(
        f"""
        <div class="result-card {result_class}">
            <p class="result-title">{result_title}</p>
            <p class="result-text">{result_desc}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.15, 1.0])

    with left_col:
        render_probability_chart(probability, threshold)

    with right_col:
        st.metric("Predicted Probability", f"{probability:.2%}")
        st.metric("Decision Threshold", f"{threshold:.2%}")
        st.metric("Model Decision", "Positive" if prediction == 1 else "Negative")

        st.markdown("### Clinical Notes")
        for note in clinical_guidance(input_values, probability):
            st.write(f"- {note}")


def render_batch_tab(model: Any, expected_features: list[str], threshold: float) -> None:
    st.subheader("Batch Inference")
    st.write("Upload a CSV file to score multiple patients and download predictions.")

    with st.expander("Required CSV columns", expanded=False):
        st.write(", ".join(expected_features))

    uploaded_file = st.file_uploader(
        "Upload patient CSV",
        type=["csv"],
        key="batch_csv",
    )

    if uploaded_file is None:
        return

    try:
        batch_raw = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Unable to read CSV file: {exc}")
        return

    missing_columns = [col for col in expected_features if col not in batch_raw.columns]
    if missing_columns:
        st.error("Missing required columns: " + ", ".join(missing_columns))
        return

    batch_input = normalize_input_frame(batch_raw[expected_features], expected_features)
    probs = model.predict_proba(batch_input)[:, 1]
    preds = (probs >= threshold).astype(int)

    output_df = batch_raw.copy()
    output_df["risk_probability"] = np.round(probs, 4)
    output_df["risk_percent"] = np.round(probs * 100.0, 2)
    output_df["prediction"] = np.where(preds == 1, "Disease Likely", "Disease Unlikely")
    output_df["threshold_used"] = threshold

    st.success(f"Scored {len(output_df)} records successfully.")
    st.dataframe(output_df.head(50), use_container_width=True)

    st.download_button(
        label="Download Scored CSV",
        data=output_df.to_csv(index=False).encode("utf-8"),
        file_name="heart_disease_batch_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_model_quality_tab(model: Any, results: dict[str, Any], threshold: float) -> None:
    st.subheader("Model Capability Dashboard")
    st.write("Core and advanced quality indicators from your saved final model outputs.")

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.metric("Accuracy", as_percent(results.get("accuracy")))
        st.metric("Precision", as_percent(results.get("precision")))
        st.metric("Balanced Accuracy", as_percent(results.get("balanced_accuracy")))
    with mcol2:
        st.metric("ROC-AUC", as_percent(results.get("roc_auc")))
        st.metric("Recall", as_percent(results.get("recall")))
        st.metric("Specificity", as_percent(results.get("specificity")))
    with mcol3:
        st.metric("F1 Score", as_percent(results.get("f1")))
        st.metric("PR-AUC", as_percent(results.get("average_precision")))
        st.metric("NPV", as_percent(results.get("npv")))
    with mcol4:
        st.metric("Brier Score", as_metric(results.get("brier_score"), digits=4))
        st.metric("MCC", as_metric(results.get("mcc"), digits=4))
        st.metric("Threshold", f"{threshold:.2f}")

    st.markdown("### Global Feature Importance")
    importance_df = extract_feature_importance(model, top_n=15)
    if importance_df.empty:
        st.info("Feature importance is unavailable for this saved model object.")
    else:
        sorted_df = importance_df.sort_values("importance", ascending=True)
        if px is not None:
            fig = px.bar(
                sorted_df,
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale=["#d8efe8", "#8ecfbf", "#0b7a75"],
            )
            fig.update_layout(
                height=480,
                margin={"l": 10, "r": 10, "t": 10, "b": 10},
                coloraxis_showscale=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"family": "Manrope, sans-serif", "color": "#1b2636"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(sorted_df.set_index("feature")["importance"])

    report_df = load_classification_report(REPORT_PATH)
    if not report_df.empty:
        st.markdown("### Per-Class Classification Report")
        st.dataframe(report_df, use_container_width=True)


def render_diagnostics_tab() -> None:
    st.subheader("Diagnostic Visual Reports")
    st.write("These charts were generated by your trained pipeline and saved as model artifacts.")

    tabs = st.tabs(list(DIAGNOSTIC_IMAGES.keys()))
    for tab, (title, image_path) in zip(tabs, DIAGNOSTIC_IMAGES.items()):
        with tab:
            if image_path.exists():
                st.image(str(image_path), caption=title, use_container_width=True)
            else:
                st.info(f"Missing file: {image_path.name}")


def main() -> None:
    st.set_page_config(
        page_title="CardioScope Model GUI",
        page_icon="ECG",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_custom_style()

    if not MODEL_PATH.exists():
        st.error(
            "Model file FINAL_MODEL.joblib was not found in the current folder. "
            "Run model.py first to generate the trained artifact."
        )
        st.stop()

    try:
        model = load_model(MODEL_PATH)
    except Exception as exc:
        st.exception(exc)
        st.stop()

    results = load_results_dict(RESULTS_PATH)
    threshold = as_float(results.get("threshold"), 0.59)
    expected_features = get_expected_features(model)

    render_hero(results, threshold)

    with st.sidebar:
        st.header("Session Controls")
        threshold = st.slider(
            "Decision threshold",
            min_value=0.10,
            max_value=0.90,
            value=float(threshold),
            step=0.01,
            help="Adjust for stricter or more sensitive decision boundaries.",
        )

        st.markdown("### Model Artifacts")
        st.write(f"Model file: {'Found' if MODEL_PATH.exists() else 'Missing'}")
        st.write(f"Results file: {'Found' if RESULTS_PATH.exists() else 'Missing'}")
        st.write(f"Report file: {'Found' if REPORT_PATH.exists() else 'Missing'}")

        st.markdown("### Guidance")
        st.write("- Use higher threshold to reduce false positives.")
        st.write("- Use lower threshold to reduce false negatives.")

    tab_predict, tab_batch, tab_quality, tab_diagnostics = st.tabs(
        [
            "Real-Time Prediction",
            "Batch Prediction",
            "Model Intelligence",
            "Diagnostics Gallery",
        ]
    )

    with tab_predict:
        render_single_prediction_tab(model, expected_features, threshold)

    with tab_batch:
        render_batch_tab(model, expected_features, threshold)

    with tab_quality:
        render_model_quality_tab(model, results, threshold)

    with tab_diagnostics:
        render_diagnostics_tab()

    st.caption(
        "Clinical safety note: this GUI is a decision-support tool. "
        "Use alongside clinician judgement, full patient history, and confirmatory testing."
    )


def ensure_streamlit_context() -> None:
    """
    Provide a clear launch instruction when script is run directly.
    """
    try:
        import streamlit.runtime as st_runtime
    except Exception:
        return

    if st_runtime.exists():
        return

    script_name = Path(__file__).name
    print("\nThis file is a Streamlit web app and must be launched with Streamlit.")
    print(f"Run: {sys.executable} -m streamlit run {script_name}\n")
    raise SystemExit(0)


if __name__ == "__main__":
    ensure_streamlit_context()
    main()
