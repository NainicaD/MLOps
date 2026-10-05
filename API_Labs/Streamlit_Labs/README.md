# Streamlit Lab - Penguin Species Predictor

Adapted from the Streamlit lab in raminmohammadi/MLOps.
A Streamlit dashboard sends penguin measurements to a FastAPI backend, which returns the predicted species.

## Changes from the original lab
- Dataset: Palmer Penguins (instead of Iris), with rows containing missing values dropped
- Model: StandardScaler + Logistic Regression pipeline (instead of a Decision Tree)
- API returns class probabilities, shown as a bar chart
- API validates that all measurements are positive

## How to run
1. `python -m venv venv`, activate it, then `pip install -r requirements.txt`
2. In `backend/`: `python train.py`, then `uvicorn main:app --reload`
3. In a second terminal, in `frontend/`: `streamlit run Dashboard.py`