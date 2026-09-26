"""
regression_model.py

Trains a Random Forest Regressor to predict delivery time (in hours)
from order-level features. This lets UrbanCart set more realistic
delivery-window promises and flag high-risk orders before dispatch.

Usage:
    python generate_data.py
    python regression_model.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


FEATURES = ["distance_km", "hub_congestion", "package_weight", "hour_of_day"]
TARGET = "delivery_time_hrs"


def load_data(path="data/deliveries.csv"):
    return pd.read_csv(path)


def train_model(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    return model, {"mae_hours": round(mae, 3), "r2_score": round(r2, 3)}


def feature_importance(model, features=FEATURES):
    return sorted(zip(features, model.feature_importances_), key=lambda x: -x[1])


if __name__ == "__main__":
    df = load_data()
    model, metrics = train_model(df)

    print("=== Model Performance ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\n=== Feature Importance ===")
    for name, score in feature_importance(model):
        print(f"{name}: {round(score, 3)}")

    # Example: predict delivery time for a new order
    sample_order = pd.DataFrame([{
        "distance_km": 12.5,
        "hub_congestion": 0.7,
        "package_weight": 3.2,
        "hour_of_day": 18,
    }])
    predicted_hours = model.predict(sample_order)[0]
    print(f"\nPredicted delivery time for sample order: {round(predicted_hours, 2)} hrs")
