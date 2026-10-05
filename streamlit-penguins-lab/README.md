# Streamlit Lab - Penguin Species Predictor

Adapted from the Streamlit lab in [raminmohammadi/MLOps](https://github.com/raminmohammadi/MLOps/tree/main/Labs/API_Labs/Streamlit_Labs).
A Streamlit dashboard sends penguin measurements to a FastAPI backend, which returns the predicted species.

## Changes from the original lab
- Dataset: Palmer Penguins (instead of Iris), including dropping rows with missing values
- Model: StandardScaler + Logistic Regression pipeline (instead of a Decision Tree)
- API returns class probabilities, shown as a bar chart in the dashboard
- Input validation on the API (all measurements must be positive)

## Project structure
```
backend/
  train.py        # trains and saves the model
  main.py         # FastAPI app: GET / (health), POST /predict
  model/          # saved model + metrics.json (created by train.py)
frontend/
  Dashboard.py    # Streamlit UI
  data/test.json  # sample input for the JSON upload option
requirements.txt
```

## How to run
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Terminal 1 - train and serve the model
cd backend
python train.py
uvicorn main:app --reload         # http://localhost:8000/docs

# Terminal 2 - start the dashboard
cd frontend
streamlit run Dashboard.py        # http://localhost:8501
```
