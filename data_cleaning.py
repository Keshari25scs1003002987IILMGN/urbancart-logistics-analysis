"""
data_cleaning.py

Week 2 task: simulates a data collection + preprocessing pipeline for
UrbanCart's raw delivery export. Injects realistic data-quality issues
into the Week 1 synthetic dataset (missing values, duplicates, outliers,
inconsistent formatting), then cleans them -- mirroring the methodology
described in the Week 2 Word report.

Usage:
    python generate_data.py       # creates the base clean dataset
    python data_cleaning.py       # dirties it, then cleans it back up
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

np.random.seed(11)


def load_base_data(path="data/deliveries.csv"):
    return pd.read_csv(path, parse_dates=["order_time", "promised_window_end", "actual_delivery"])


def simulate_raw_export(df: pd.DataFrame) -> pd.DataFrame:
    """Reintroduces realistic data-quality problems on top of the clean base data."""
    dirty = df.copy()

    # 1. Missing values in a few numeric columns
    for col in ["trip_cost", "hub_congestion", "package_weight"]:
        mask = np.random.rand(len(dirty)) < 0.03
        dirty.loc[mask, col] = np.nan

    # 2. Duplicate rows (same order logged twice by two systems)
    dup_sample = dirty.sample(frac=0.02, random_state=1)
    dirty = pd.concat([dirty, dup_sample], ignore_index=True)

    # 3. Outliers in distance_km and trip_cost
    outlier_idx = dirty.sample(frac=0.01, random_state=2).index
    dirty.loc[outlier_idx, "distance_km"] *= 8
    dirty.loc[outlier_idx, "trip_cost"] *= 6

    # 4. Inconsistent formatting in zone_id (mixed case / stray whitespace)
    dirty["zone_id"] = dirty["zone_id"].astype(str)
    messy_idx = dirty.sample(frac=0.1, random_state=3).index
    dirty.loc[messy_idx, "zone_id"] = " zone" + dirty.loc[messy_idx, "zone_id"]

    return dirty


def cap_outliers_iqr(series: pd.Series, factor=1.5) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - factor * iqr, q3 + factor * iqr
    return series.clip(lower=lower, upper=upper)


def clean_pipeline(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    # Handle missing values (median imputation for skewed numeric fields)
    for col in ["trip_cost", "hub_congestion", "package_weight"]:
        df[col] = df[col].fillna(df[col].median())

    # Remove duplicates
    df = df.drop_duplicates(subset="order_id", keep="first")

    # Cap outliers
    df["distance_km"] = cap_outliers_iqr(df["distance_km"])
    df["trip_cost"] = cap_outliers_iqr(df["trip_cost"])

    # Standardize formats
    df["zone_id"] = df["zone_id"].astype(str).str.strip().str.lower().str.replace("zone", "", regex=False)
    df["order_time"] = pd.to_datetime(df["order_time"], errors="coerce")

    return df


def normalize_features(df: pd.DataFrame, cols=("distance_km", "package_weight", "trip_cost")) -> pd.DataFrame:
    df = df.copy()
    scaler = MinMaxScaler()
    df[list(cols)] = scaler.fit_transform(df[list(cols)])
    return df


if __name__ == "__main__":
    base = load_base_data()

    raw = simulate_raw_export(base)
    print("=== Simulated Raw Export ===")
    print(f"Rows: {len(raw)} (includes injected duplicates)")
    print(f"Missing values per column:\n{raw.isna().sum()[raw.isna().sum() > 0]}")
    print(f"Duplicate order_ids: {raw['order_id'].duplicated().sum()}")

    cleaned = clean_pipeline(raw)
    print("\n=== After Cleaning ===")
    print(f"Rows: {len(cleaned)}")
    print(f"Remaining missing values: {cleaned.isna().sum().sum()}")
    print(f"Remaining duplicate order_ids: {cleaned['order_id'].duplicated().sum()}")
    print(f"Distance range after outlier capping: "
          f"{cleaned['distance_km'].min():.2f} - {cleaned['distance_km'].max():.2f} km")

    normalized = normalize_features(cleaned)
    normalized.to_csv("data/cleaned_deliveries.csv", index=False)
    print("\nSaved normalized, cleaned dataset to data/cleaned_deliveries.csv")
