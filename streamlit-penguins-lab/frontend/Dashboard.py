"""Streamlit dashboard that sends penguin measurements to the FastAPI backend."""
import json

import pandas as pd
import requests
import streamlit as st

API_URL = "http://localhost:8000"

# label, min, max, default, step
FEATURE_SLIDERS = {
    "bill_length_mm": ("Bill length (mm)", 32.0, 60.0, 44.0, 0.1),
    "bill_depth_mm": ("Bill depth (mm)", 13.0, 22.0, 17.0, 0.1),
    "flipper_length_mm": ("Flipper length (mm)", 170.0, 232.0, 200.0, 1.0),
    "body_mass_g": ("Body mass (g)", 2700.0, 6300.0, 4200.0, 50.0),
}

st.set_page_config(page_title="Penguin Species Predictor", page_icon="🐧")


def backend_online() -> bool:
    try:
        return requests.get(f"{API_URL}/", timeout=3).status_code == 200
    except requests.exceptions.RequestException:
        return False


# ---------- Sidebar: status + inputs ----------
with st.sidebar:
    st.header("Backend status")
    if backend_online():
        st.success("FastAPI backend is online")
    else:
        st.error("Backend offline. Start it with `uvicorn main:app --reload` in /backend")

    st.header("Input")
    mode = st.radio("Choose input method", ["Sliders", "Upload JSON"])

    payload = None
    if mode == "Sliders":
        payload = {
            name: st.slider(label, lo, hi, default, step)
            for name, (label, lo, hi, default, step) in FEATURE_SLIDERS.items()
        }
    else:
        uploaded = st.file_uploader("Upload a JSON file", type="json")
        if uploaded is not None:
            payload = json.load(uploaded)
            st.json(payload)

    predict_clicked = st.button("Predict", type="primary")

# ---------- Main page ----------
st.title("🐧 Penguin Species Predictor")
st.write(
    "Enter a penguin's measurements in the sidebar and click **Predict**. "
    "The dashboard sends them to a FastAPI backend running a Logistic Regression "
    "model trained on the Palmer Penguins dataset."
)

if predict_clicked:
    if payload is None:
        st.warning("Upload a JSON file first.")
    else:
        try:
            with st.spinner("Asking the model..."):
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
            response.raise_for_status()
            result = response.json()

            st.success(f"Predicted species: **{result['species']}**")
            st.subheader("Class probabilities")
            st.bar_chart(pd.Series(result["probabilities"], name="probability"))
        except requests.exceptions.ConnectionError:
            st.error("Could not reach the backend. Is uvicorn running?")
        except requests.exceptions.HTTPError:
            st.error(f"The API rejected the input: {response.text}")
