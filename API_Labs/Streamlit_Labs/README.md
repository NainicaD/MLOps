## How to run

All commands start from the `Streamlit_Labs` folder. You'll need two terminal windows: one for the backend (the model API) and one for the frontend (the dashboard).

### 1. Set up (first time only)

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

- Mac/Linux: `source venv/bin/activate`
- Windows: `venv\Scripts\activate`

You should see `(venv)` at the start of your terminal prompt. Then install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Train the model and start the backend (Terminal 1)

```bash
cd backend
python train.py
uvicorn main:app --reload
```

- `train.py` trains the model, prints its accuracy, and saves it to `backend/model/`.
- `uvicorn` starts the API at http://127.0.0.1:8000. Keep this terminal open.
- Optional: open http://127.0.0.1:8000/docs to test the `/predict` endpoint in your browser.

### 3. Start the dashboard (Terminal 2)

Open a second terminal, go to the `Streamlit_Labs` folder, and activate the virtual environment again (see step 1). Then run:

```bash
cd frontend
streamlit run Dashboard.py
```

The dashboard opens at http://localhost:8501.

### 4. Use the dashboard

1. Check that the sidebar shows **"FastAPI backend is online"**.
2. Enter penguin measurements with the sliders, or upload `frontend/data/test.json`.
3. Click **Predict** to see the predicted species and the probability for each class.

### To stop

Press `Ctrl+C` in each terminal.