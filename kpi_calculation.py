"""
kpi_calculation.py

Computes the three primary KPIs for UrbanCart's logistics operation:
  1. On-Time Delivery Rate (OTDR)
  2. Average Delivery Cost per Order
  3. Order Fulfillment Cycle Time

Usage:
    python generate_data.py     # creates data/deliveries.csv (run once)
    python kpi_calculation.py
"""

import pandas as pd


def load_data(path="data/deliveries.csv"):
    df = pd.read_csv(path, parse_dates=["order_time", "promised_window_end", "actual_delivery"])
    return df


def compute_kpis(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["on_time"] = df["actual_delivery"] <= df["promised_window_end"]

    otdr = df["on_time"].mean() * 100
    avg_cost_per_order = df["trip_cost"].mean()
    cycle_time = (df["actual_delivery"] - df["order_time"]).mean()

    # secondary KPI: orders delivered per unique route (proxy for vehicle utilization)
    route_utilization = df.groupby("route_id")["order_id"].count().mean()

    return {
        "on_time_delivery_rate_pct": round(otdr, 2),
        "avg_delivery_cost_per_order": round(avg_cost_per_order, 2),
        "avg_fulfillment_cycle_time": str(cycle_time),
        "avg_orders_per_route": round(route_utilization, 2),
    }


def kpis_by_zone(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["on_time"] = df["actual_delivery"] <= df["promised_window_end"]
    grouped = df.groupby("zone_id").agg(
        otdr_pct=("on_time", lambda s: round(s.mean() * 100, 2)),
        avg_cost=("trip_cost", lambda s: round(s.mean(), 2)),
        orders=("order_id", "count"),
    ).reset_index()
    return grouped.sort_values("otdr_pct")


if __name__ == "__main__":
    df = load_data()
    kpis = compute_kpis(df)

    print("=== Overall KPIs ===")
    for k, v in kpis.items():
        print(f"{k}: {v}")

    print("\n=== Worst 5 zones by On-Time Delivery Rate ===")
    print(kpis_by_zone(df).head(5).to_string(index=False))
