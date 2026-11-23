from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.joblib"
LABELS = ["Minnow", "Dolphin", "Whale"]
FEATURE_COLS = ["Age", "Country", "Device", "GameGenre", "SessionCount", "AverageSessionLength"]

st.set_page_config(
    page_title="NovaPlay | Player Segment Studio",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --ink:#17243b; --muted:#43536b; --panel:#ffffff; --line:#c9d5e7; --blue:#2348c8; --teal:#006f78; --violet:#6039a6; }
.stApp { background:linear-gradient(145deg,#f7f9fd 0%,#edf3fc 58%,#e8f3f7 100%); color:var(--ink); font-family:'DM Sans',sans-serif; }
[data-testid="stAppViewContainer"], [data-testid="stHeader"] { background:transparent; }
.block-container { max-width:1220px; padding-top:2rem; padding-bottom:3rem; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.025em; color:#17243b; }
p, label, [data-testid="stWidgetLabel"] p, [data-testid="stMarkdownContainer"] { color:#263750; }
.hero { border:1px solid #d5c7f1; border-left:7px solid #6d28d9; border-radius:18px; padding:18px 26px; background:linear-gradient(110deg,#f2edff,#fbfaff 70%,#fff1f3); box-shadow:0 8px 22px rgba(60,38,106,.08); margin-bottom:20px; }
.eyebrow { color:#5b21b6; font-weight:700; letter-spacing:.12em; text-transform:uppercase; font-size:.72rem; }
.hero h1 { font-size:2rem; margin:.3rem 0 0; color:#17243b; }
.panel { border:1px solid var(--line); border-radius:18px; padding:20px 22px; background:#fff; min-height:116px; box-shadow:0 6px 18px rgba(36,57,93,.05); }
.offer-card { border-left:6px solid #6d28d9; }
.goal-card { margin-top:12px; border:1px solid #d8c8f3; border-radius:16px; padding:16px 18px; background:#f4efff; }
.panel-label { color:#43536b; font-size:.78rem; text-transform:uppercase; letter-spacing:.09em; font-weight:700; }
.panel-value { color:#17243b; font-family:'Space Grotesk',sans-serif; font-size:1.45rem; font-weight:700; margin-top:8px; }
.segment { border:2px solid #5970b8; border-radius:20px; padding:26px; background:linear-gradient(140deg,#e3f7f3,#edf0ff); }
.segment h2 { font-size:2rem; margin:.3rem 0; color:#17243b; }
.small { color:#43536b; font-size:.92rem; }
div[data-testid="stForm"] { border:1px solid var(--line); border-radius:18px; padding:20px 22px; background:#fff; box-shadow:0 8px 24px rgba(36,57,93,.06); }
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label { color:#243751 !important; font-weight:600 !important; }
div[data-testid="stNumberInput"] input, div[data-testid="stSelectbox"] [data-baseweb="select"] > div { background:#fff !important; color:#17243b !important; border-color:#9aabc4 !important; }
div[data-testid="stNumberInput"] button { color:#243751 !important; }
button[data-testid="stBaseButton-primary"], div[data-testid="stFormSubmitButton"] button { border:2px solid #5420aa !important; border-radius:14px !important; color:#fff !important; font-weight:800 !important; background:#6d28d9 !important; padding:.78rem 1.2rem !important; box-shadow:0 5px 0 #4c1d95,0 8px 18px rgba(83,42,151,.2) !important; transition:transform .15s ease,background .15s ease !important; }
button[data-testid="stBaseButton-primary"] *, div[data-testid="stFormSubmitButton"] button * { color:#fff !important; }
button[data-testid="stBaseButton-primary"]:hover, div[data-testid="stFormSubmitButton"] button:hover { color:#fff !important; border-color:#42168d !important; background:#5720b5 !important; transform:translateY(-1px); }
button[data-testid="stBaseButton-primary"]:focus-visible, div[data-testid="stFormSubmitButton"] button:focus-visible { outline:3px solid #006f78 !important; outline-offset:3px !important; }
hr { border-color:#c9d5e7; }
[data-testid="stMetric"] { background:#fff; padding:14px 16px; border:1px solid var(--line); border-radius:14px; box-shadow:0 5px 14px rgba(36,57,93,.04); }
[data-testid="stMetricLabel"] p { color:#43536b !important; }
[data-testid="stMetricValue"] { color:#17243b !important; }
div[data-testid="stAlert"] { border-radius:14px; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("model.joblib is missing. Run train_model.py before starting the app.")
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as exc:
    st.error(f"The prediction model could not be loaded: {exc}")
    st.stop()

st.markdown("""
<div class="hero">
  <div class="eyebrow">NOVAPLAY STUDIOS · PLAYER INTELLIGENCE</div>
  <h1>Player Segment Lab</h1>
</div>
""", unsafe_allow_html=True)

country_options = [
    "Australia", "Bangladesh", "Brazil", "Canada", "China", "Denmark", "Egypt", "France", "Germany", "India", "Iran", "Italy", "Japan", "Mexico", "Netherlands", "Norway", "Russia", "Saudi Arabia", "South Korea", "Spain", "Sri Lanka", "Sweden", "Switzerland", "Turkey", "United Kingdom", "United States", "Afghanistan", "Unknown"
]
genre_options = ["Action RPG", "Adventure", "Battle Royale", "Card", "Casual", "Fighting", "MMORPG", "MOBA", "Puzzle", "Racing", "Role Playing", "Sandbox", "Simulation", "Sports", "Strategy", "Unknown"]

col_form, col_side = st.columns([1.15, .85], gap="large")
with col_form:
    st.subheader("Player profile")
    st.write("Use the profile information available at the intended prediction time.")
    with st.form("player_profile", clear_on_submit=False):
        a, b = st.columns(2)
        with a:
            age = st.number_input("Age", min_value=13, max_value=100, value=30, step=1, help="Enter an age from 13 to 100.")
            country = st.selectbox("Country", country_options, index=country_options.index("United States"))
            device = st.selectbox("Device", ["Android", "iOS", "Unknown"])
        with b:
            genre = st.selectbox("Favourite game genre", genre_options, index=genre_options.index("Adventure"))
            sessions = st.number_input("Session count", min_value=0, max_value=500, value=10, step=1, help="Number of sessions in the selected observation period.")
            avg_length = st.number_input("Average session length (minutes)", min_value=0.0, max_value=600.0, value=20.0, step=1.0, help="Use a non-negative value in minutes.")
        submitted = st.form_submit_button("Predict player segment", use_container_width=True)

record = None
prediction = None
scores = {}
score = 0.0
if submitted:
    record = pd.DataFrame([{
        "Age": float(age),
        "Country": str(country),
        "Device": str(device),
        "GameGenre": str(genre),
        "SessionCount": float(sessions),
        "AverageSessionLength": float(avg_length),
    }], columns=FEATURE_COLS)
    prediction = str(model.predict(record)[0])
    probabilities = model.predict_proba(record)[0]
    classes = list(model.named_steps["model"].classes_)
    scores = {str(label): float(prob) for label, prob in zip(classes, probabilities)}
    score = scores.get(prediction, 0.0)

with col_form:
    if submitted:
        ideas = {
            "Minnow": ("Starter bundle", "Test an optional, low-price starter bundle with clear value."),
            "Dolphin": ("Value bundle", "Test a time-limited bundle or loyalty reward suited to regular players."),
            "Whale": ("Premium experience", "Test a premium bundle or VIP-style benefit with clear value."),
        }
        offer_title, offer_text = ideas[prediction]
        st.markdown(f"""
        <div class="panel offer-card">
          <div class="panel-label">OFFER IDEA FOR THIS SEGMENT</div>
          <div class="panel-value">{offer_title}</div>
          <div class="small">{offer_text}</div>
        </div>
        <div class="goal-card">
          <div class="panel-label">HOW THIS SUPPORTS NOVAPLAY'S GOAL</div>
          <div class="small">Use the predicted segment to define a test audience. Compare this offer with a holdout group using incremental IAP revenue, contribution margin and retention. Expand only if results support progress toward the 15% revenue-growth target.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="panel offer-card">
          <div class="panel-label">OFFER IDEA + BUSINESS TEST</div>
          <div class="panel-value">Will appear after prediction</div>
          <div class="small">The app will suggest an offer for the predicted segment and show how NovaPlay could test its business value.</div>
        </div>
        """, unsafe_allow_html=True)

with col_side:
    st.subheader("Player segment result")
    if not submitted:
        st.markdown("""
        <div class="segment">
          <div class="eyebrow">YOUR RESULT WILL APPEAR HERE</div>
          <h2>Ready when you are</h2>
          <div class="small">Complete the player profile and select the purple button to see the segment and a tailored offer idea.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="segment">
          <div class="eyebrow">PREDICTED SPENDING SEGMENT</div>
          <h2>{prediction}</h2>
        </div>
        """, unsafe_allow_html=True)
        st.metric("Model score for this segment", f"{score:.0%}")
        st.progress(score)
        st.caption("This is the model's score for this input—not the model's overall accuracy, and not a guarantee of future spending.")
        with st.expander("Compare scores across all segments"):
            st.bar_chart(pd.DataFrame({"Model score": [scores.get(label, 0.0) for label in LABELS]}, index=LABELS), horizontal=True, height=170, color="#6d28d9")

st.info("The model uses age, country, device, favourite game genre, session count and average session length. Spending amounts and generated 30-day helper fields are not entered.")

if submitted:
    st.markdown("#### Why did the model predict this?")
    st.write("This chart ranks the features the fitted tree used most across its training data. It helps explain the model overall; it does not identify the exact reason for this one player's result, and it does not show cause and effect.")
    preprocessor = model.named_steps["preprocess"]
    estimator = model.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    raw_importances = pd.Series(estimator.feature_importances_, index=names)
    aggregated = {}
    for name, importance in raw_importances.items():
        if name.startswith("numeric__"):
            base = name.split("__", 1)[1]
        elif name.startswith("categorical__"):
            base = name.split("__", 1)[1].split("_", 1)[0]
        else:
            base = name
        aggregated[base] = aggregated.get(base, 0.0) + float(importance)
    importance = pd.Series(aggregated).sort_values(ascending=True).tail(6)
    st.bar_chart(importance, horizontal=True, height=250, color="#6d28d9")
    st.caption("Feature importance is a global summary, not an explanation of an individual prediction.")

    with st.expander("View the player inputs used"):
        st.dataframe(record, hide_index=True, use_container_width=True)

st.markdown("---")
st.markdown("#### Project evaluation reference")
m1, m2, m3, m4 = st.columns(4)
for col, label, value in [
    (m1, "Test accuracy", "94.62%"),
    (m2, "Balanced accuracy", "86.23%"),
    (m3, "Macro-F1", "86.65%"),
    (m4, "Whale precision", "81.05%"),
]:
    with col:
        st.metric(label, value)
st.caption("These are the team's reported held-out test results for the Gini Decision Tree. They describe the project dataset, not guaranteed performance on live NovaPlay players.")

with st.expander("What this demo can—and cannot—show"):
    st.write("This app demonstrates how a player profile can be assigned to a spending segment and how that segment might inform a promotion experiment. It does not estimate how much the player will spend, determine the best offer, or prove that a promotion will increase revenue. A real campaign decision should be tested against a holdout group and assessed using incremental revenue, contribution margin, conversion and retention.")

st.caption("NovaPlay Studios · Academic project demonstration · Model: Gini Decision Tree")
