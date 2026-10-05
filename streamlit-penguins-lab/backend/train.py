"""Train a Logistic Regression model on the Palmer Penguins dataset and save it."""
from pathlib import Path
import json

import joblib
from palmerpenguins import load_penguins
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
TARGET = "species"
MODEL_DIR = Path(__file__).parent / "model"


def main():
    # 1. Load data and drop rows with missing measurements
    df = load_penguins()[FEATURES + [TARGET]].dropna()
    X, y = df[FEATURES], df[TARGET]

    # 2. Split (stratified so all three species appear in both sets)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Scaling + Logistic Regression in one pipeline,
    #    so the API never has to scale inputs separately
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)

    # 4. Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.3f}\n")
    print(classification_report(y_test, preds))

    # 5. Save model + metrics
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "penguin_model.joblib")
    metrics = {"accuracy": round(acc, 4), "n_train": len(X_train), "n_test": len(X_test)}
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved model to {MODEL_DIR / 'penguin_model.joblib'}")


if __name__ == "__main__":
    main()
