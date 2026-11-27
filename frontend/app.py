import os
import requests
import pandas as pd
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Iris Species Classifier | ML System",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F3F4F6 0%, #E5E7EB 100%);
        border-radius: 12px;
        padding: 1.2rem;
        border-left: 6px solid #3B82F6;
        margin-top: 1rem;
    }
    .pred-highlight {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1D4ED8;
    }
</style>
""", unsafe_allow_html=True)

# 2. Sidebar Configuration
st.sidebar.title("⚙️ ML System Settings")

# API URL Configuration (Docker network compatible)
DEFAULT_API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
api_url = st.sidebar.text_input("FastAPI Backend URL", value=DEFAULT_API_URL).rstrip("/")

# Health check test
api_online = False
try:
    health_resp = requests.get(f"{api_url}/", timeout=2)
    if health_resp.status_code == 200:
        api_online = True
        st.sidebar.success("Backend: Connected (HTTP 200)")
    else:
        st.sidebar.warning(f"Backend Warning: Status {health_resp.status_code}")
except Exception:
    st.sidebar.error("Backend: Disconnected (FastAPI not reachable)")

st.sidebar.markdown("---")
st.sidebar.subheader("📌 Preset Sample Values")
st.sidebar.caption("Click any preset below to populate the input sliders with realistic flower features:")

PRESETS = {
    "🌸 Iris Setosa": {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
    "🌿 Iris Versicolor": {"sepal_length": 5.9, "sepal_width": 3.0, "petal_length": 4.2, "petal_width": 1.5},
    "🌺 Iris Virginica": {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3}
}

for name, values in PRESETS.items():
    if st.sidebar.button(name, use_container_width=True):
        for k, v in values.items():
            st.session_state[f"input_{k}"] = v

st.sidebar.markdown("---")
st.sidebar.info("""
**SIS - 3 ML Pipeline Design**
* **Frontend:** Streamlit
* **API Service:** FastAPI
* **Lifecycle:** MLflow Tracking & Registry
* **Containerization:** Docker Compose
""")

# 3. Main Dashboard Layout
st.markdown('<div class="main-title">🌸 Iris Flower Species Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Interactive frontend communicating with containerized FastAPI backend and tracked via MLflow.</div>', unsafe_allow_html=True)

# Tabs for Organization
tab1, tab2, tab3 = st.tabs(["🎯 Single Prediction", "📊 Batch Evaluation", "📈 MLflow & Pipeline Architecture"])

# ----------------- TAB 1: Single Prediction -----------------
with tab1:
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("1. Input Feature Measurements (cm)")

        sl = st.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0,
                       value=float(st.session_state.get("input_sepal_length", 5.1)), step=0.1, key="sl_slider")
        sw = st.slider("Sepal Width (cm)", min_value=2.0, max_value=4.5,
                       value=float(st.session_state.get("input_sepal_width", 3.5)), step=0.1, key="sw_slider")
        pl = st.slider("Petal Length (cm)", min_value=1.0, max_value=7.0,
                       value=float(st.session_state.get("input_petal_length", 1.4)), step=0.1, key="pl_slider")
        pw = st.slider("Petal Width (cm)", min_value=0.1, max_value=2.5,
                       value=float(st.session_state.get("input_petal_width", 0.2)), step=0.1, key="pw_slider")

        predict_btn = st.button("🚀 Predict Species", type="primary", use_container_width=True)

    with col_result:
        st.subheader("2. Prediction & Confidence")

        if predict_btn:
            payload = {
                "sepal_length": sl,
                "sepal_width": sw,
                "petal_length": pl,
                "petal_width": pw
            }

            try:
                with st.spinner("Querying FastAPI backend..."):
                    resp = requests.post(f"{api_url}/predict", json=payload, timeout=5)

                if resp.status_code == 200:
                    data = resp.json()
                    pred_species = data.get("prediction", "Unknown").capitalize()
                    class_id = data.get("class_id", -1)
                    probs = data.get("probabilities", {})

                    emoji_map = {
                        "Setosa": "🌸",
                        "Versicolor": "🌿",
                        "Virginica": "🌺"
                    }
                    badge = emoji_map.get(pred_species, "🌼")

                    st.markdown(f"""
                    <div class="metric-card">
                        <div>Predicted Class:</div>
                        <div class="pred-highlight">{badge} Iris {pred_species}</div>
                        <div style="color: #6B7280; font-size: 0.9rem; margin-top: 4px;">Class Index: {class_id}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    if probs:
                        st.markdown("#### Probability Distribution:")
                        df_probs = pd.DataFrame({
                            "Species": [f"{emoji_map.get(k.capitalize(), '')} {k.capitalize()}" for k in probs.keys()],
                            "Confidence": list(probs.values())
                        }).set_index("Species")

                        st.bar_chart(df_probs, height=220)

                        top_species = pred_species.lower()
                        top_conf = probs.get(top_species, 0.0) * 100
                        st.info(f"Model Confidence for **Iris {pred_species}**: **{top_conf:.2f}%**")

                    with st.expander("🔍 View Raw JSON Response"):
                        st.json(data)
                else:
                    st.error(f"API Error ({resp.status_code}): {resp.text}")

            except requests.exceptions.ConnectionError:
                st.error(f"Cannot reach API at `{api_url}`. Ensure FastAPI is running on port 8000.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.info("Adjust the sliders on the left or pick a sample preset from the sidebar, then click **Predict Species**.")


# ----------------- TAB 2: Batch Evaluation -----------------
with tab2:
    st.subheader("Batch Sample Testing")
    st.write("Test multiple pre-configured samples against the API in one run:")

    sample_table = pd.DataFrame([
        {"Sample": "Sample #1", "sepal_length": 5.0, "sepal_width": 3.6, "petal_length": 1.4, "petal_width": 0.2, "True Species": "setosa"},
        {"Sample": "Sample #2", "sepal_length": 5.7, "sepal_width": 2.8, "petal_length": 4.1, "petal_width": 1.3, "True Species": "versicolor"},
        {"Sample": "Sample #3", "sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5, "True Species": "virginica"},
        {"Sample": "Sample #4", "sepal_length": 6.5, "sepal_width": 3.0, "petal_length": 5.5, "petal_width": 1.8, "True Species": "virginica"}
    ])

    st.dataframe(sample_table, use_container_width=True)

    if st.button("Run Batch Prediction Test", use_container_width=True):
        results = []
        for _, row in sample_table.iterrows():
            payload = {
                "sepal_length": float(row["sepal_length"]),
                "sepal_width": float(row["sepal_width"]),
                "petal_length": float(row["petal_length"]),
                "petal_width": float(row["petal_width"])
            }
            try:
                res = requests.post(f"{api_url}/predict", json=payload, timeout=3)
                if res.status_code == 200:
                    pred = res.json().get("prediction")
                    results.append(pred)
                else:
                    results.append("Error")
            except Exception:
                results.append("Unavailable")

        sample_table["Model Prediction"] = results
        sample_table["Match"] = sample_table["True Species"] == sample_table["Model Prediction"]
        st.dataframe(sample_table, use_container_width=True)


# ----------------- TAB 3: MLflow & Lifecycle Architecture -----------------
with tab3:
    st.subheader("System Architecture & MLflow Integration")
    st.markdown("""
    #### MLflow Integration Summary
    * **Experiment:** `iris_classification_experiment`
    * **Parameters Logged:** `n_estimators`, `max_depth`, `criterion`, `random_state`, `test_size`
    * **Metrics Logged:** `accuracy`, `f1_score_macro`, `f1_score_weighted`, `precision_macro`, `recall_macro`
    * **Artifacts Logged:** `model.joblib`, `confusion_matrix.png`, `classification_report.txt`
    * **Model Registry:** Model `IrisClassifier` registered with version tracking and production tags.
    """)