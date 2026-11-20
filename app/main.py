"""Streamlit demonstration for the mobile-game spending-segment classifier."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.train_model import (  # noqa: E402
    CATEGORICAL_FEATURES,
    CLASS_ORDER,
    FEATURES,
    NUMERIC_FEATURES,
    SOURCE_FILE,
    canonicalize_categories,
    load_source,
)

MODEL_FILE = ROOT / "models" / "spending_segment_model.joblib"
META_FILE = ROOT / "models" / "model_metadata.json"

st.set_page_config(page_title="Player spending segment demo", page_icon="🎮", layout="wide")
st.title("Mobile game player spending segments")
st.caption("A course project demonstration for NovaPlay Studios")

st.warning(
    "Educational demonstration only. The dataset publisher describes the source data as synthetic, "
    "and the expanded records are generated. Predictions are not validated for real players and "
    "must not be used to target or treat individuals. Classification scores do not show that an "
    "offer increases revenue."
)

if not MODEL_FILE.exists() or not META_FILE.exists():
    st.error("The model files are not available yet. Run `python src/train_model.py` from the project root.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

@st.cache_data
def load_metadata():
    return json.loads(META_FILE.read_text(encoding="utf-8"))

@st.cache_data
def get_options():
    source = load_source()
    options = {}
    for column in CATEGORICAL_FEATURES:
        values = sorted(source[column].dropna().astype(str).unique().tolist())
        options[column] = values or ["Unknown"]
    return options

model = load_model()
metadata = load_metadata()

with st.expander("About the model and its evaluation", expanded=True):
    st.write(
        "The model predicts one of three labels using age, session count, average session length, "
        "gender, country, device, and game genre. Purchase amount and other purchase outcome fields "
        "are excluded to reduce direct target leakage. Preprocessing is fitted inside the training "
        "pipeline. Two model families were compared with stratified cross-validation and a held-out "
        "test set."
    )
    st.write(f"Selected model: **{metadata['selected_model']}**")
    st.write(
        f"Evaluation set: {metadata['evaluation_rows']:,} records from the original "
        f"{metadata['source_rows_used']:,}-row source dataset."
    )
    results = pd.DataFrame(metadata["comparison"])
    show_columns = ["model", "test_macro_f1", "test_balanced_accuracy", "test_accuracy", "whale_recall"]
    st.dataframe(results[show_columns].round(3), hide_index=True, use_container_width=True)
    selected_result = next(
        row for row in metadata["comparison"] if row["model"] == metadata["selected_model"]
    )
    if selected_result["whale_recall"] < 0.5:
        st.error(
            f"Whale recall on the held-out test set is low ({selected_result['whale_recall']:.1%}). "
            "The model misses many Whale cases and is not suitable for operational targeting."
        )
    st.write(
        "The source dataset is described by its publisher as synthetic. The 250,000-row expanded "
        "file is not used for model evaluation because it contains generated observations. These "
        "metrics should be read as a classroom workflow demonstration, not as validated business performance."
    )

st.subheader("Try one hypothetical profile")
st.caption("Use fictional inputs only. Do not enter identifiable or sensitive player information.")
options = get_options()
with st.form("single_prediction"):
    a, b, c = st.columns(3)
    with a:
        age = st.number_input("Age", min_value=13, max_value=100, value=25, step=1)
        session_count = st.number_input("Session count", min_value=0, max_value=10000, value=20, step=1)
        duration = st.number_input("Average session length (minutes)", min_value=0.0, max_value=600.0, value=15.0, step=1.0)
    with b:
        gender = st.selectbox("Gender", options["Gender"])
        country = st.selectbox("Country", options["Country"])
        device = st.selectbox("Device", options["Device"])
    with c:
        genre = st.selectbox("Game genre", options["GameGenre"])
    submitted = st.form_submit_button("Estimate segment", type="primary")

if submitted:
    row = pd.DataFrame([{
        "Age": age,
        "SessionCount": session_count,
        "AverageSessionLength": duration,
        "Gender": gender,
        "Country": country,
        "Device": device,
        "GameGenre": genre,
    }], columns=FEATURES)
    label = model.predict(row)[0]
    st.info(f"Model estimate: **{label}**")
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(row)[0]
        probability_frame = pd.DataFrame({"Segment": model.classes_, "Model score": probabilities})
        probability_frame = probability_frame.set_index("Segment").reindex(CLASS_ORDER).dropna()
        st.bar_chart(probability_frame)
        st.caption("Scores are model probabilities for this fitted classifier, not guarantees or customer value estimates.")

st.divider()
st.subheader("Score a CSV file")
st.write("Upload a CSV with these feature columns: " + ", ".join(f"`{x}`" for x in FEATURES) + ".")
upload = st.file_uploader("Choose a CSV", type=["csv"])
if upload is not None:
    try:
        batch = pd.read_csv(upload)
        missing = [column for column in FEATURES if column not in batch.columns]
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        elif batch.empty:
            st.error("The uploaded CSV has no rows.")
        else:
            clean = canonicalize_categories(batch[FEATURES].copy())
            for column in NUMERIC_FEATURES:
                clean[column] = pd.to_numeric(clean[column], errors="coerce")
            predicted = model.predict(clean)
            result = batch.copy()
            result["PredictedSpendingSegment"] = predicted
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(clean)
                for index, label in enumerate(model.classes_):
                    result[f"Score_{label}"] = probabilities[:, index]
            st.success(f"Scored {len(result):,} rows.")
            st.dataframe(result.head(50), use_container_width=True)
            st.download_button(
                "Download predictions CSV",
                result.to_csv(index=False).encode("utf-8"),
                file_name="player_segment_predictions.csv",
                mime="text/csv",
            )
            st.caption("Uploaded data is processed in this app session; avoid sensitive or identifiable records.")
    except Exception as exc:
        st.error(f"The CSV could not be processed: {exc}")

st.divider()
st.subheader("Responsible business use")
st.write(
    "A spending-segment classifier cannot identify which offer causes a profitable response. "
    "Before any real campaign, validate the model on a representative, time-separated dataset, "
    "review privacy and fairness risks, and test offers with a randomized holdout. Compare "
    "incremental revenue after offer cost, contribution margin, and player retention."
)
