"""
generate_data.py

Creates two synthetic CSV datasets that stand in for UrbanCart's real
order-management and fleet-tracking exports:

    data/deliveries.csv   -> order-level delivery records
    data/zones.csv        -> pincode/zone-level aggregated stats

Run this first; every other script in the repo reads these files.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

N_ORDERS = 2000
N_ZONES = 30

OUT_DIR = "data"
import os
os.makedirs(OUT_DIR, exist_ok=True)


def generate_deliveries(n=N_ORDERS):
    zone_ids = np.random.randint(1, N_ZONES + 1, size=n)
    distance_km = np.round(np.random.gamma(shape=3.0, scale=2.5, size=n), 2)
    hub_congestion = np.round(np.random.uniform(0, 1, size=n), 2)  # 0 = free flow, 1 = gridlock
    package_weight = np.round(np.random.gamma(shape=2.0, scale=1.5, size=n), 2)
    hour_of_day = np.random.randint(8, 22, size=n)

    base_time = 0.5 + 0.15 * distance_km + 1.8 * hub_congestion + 0.05 * package_weight
    noise = np.random.normal(0, 0.4, size=n)
    delivery_time_hrs = np.clip(base_time + noise, 0.3, None)

    order_time = [
        datetime(2026, 8, 24) + timedelta(days=int(d), hours=int(h))
        for d, h in zip(np.random.randint(0, 28, size=n), hour_of_day)
    ]
    actual_delivery = [
        ot + timedelta(hours=float(dt)) for ot, dt in zip(order_time, delivery_time_hrs)
    ]
    promised_window_hrs = 3.0
    promised_window_end = [ot + timedelta(hours=promised_window_hrs) for ot in order_time]

    trip_cost = np.round(40 + 12 * distance_km + 15 * hub_congestion + np.random.normal(0, 8, size=n), 2)
    route_id = np.random.randint(1, 400, size=n)

    df = pd.DataFrame({
        "order_id": np.arange(1, n + 1),
        "zone_id": zone_ids,
        "route_id": route_id,
        "distance_km": distance_km,
        "hub_congestion": hub_congestion,
        "package_weight": package_weight,
        "hour_of_day": hour_of_day,
        "order_time": order_time,
        "promised_window_end": promised_window_end,
        "delivery_time_hrs": np.round(delivery_time_hrs, 2),
        "actual_delivery": actual_delivery,
        "trip_cost": trip_cost,
    })
    return df


def generate_zones(deliveries: pd.DataFrame, n_zones=N_ZONES):
    rows = []
    for zid in range(1, n_zones + 1):
        sub = deliveries[deliveries["zone_id"] == zid]
        if sub.empty:
            continue
        delay_mins = ((sub["actual_delivery"] - sub["promised_window_end"]).dt.total_seconds() / 60).clip(lower=0)
        rows.append({
            "zone_id": zid,
            "avg_delay_mins": round(delay_mins.mean(), 2),
            "cost_per_order": round(sub["trip_cost"].mean(), 2),
            "order_density": len(sub),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    deliveries = generate_deliveries()
    deliveries.to_csv(f"{OUT_DIR}/deliveries.csv", index=False)

    zones = generate_zones(deliveries)
    zones.to_csv(f"{OUT_DIR}/zones.csv", index=False)

    print(f"Wrote {len(deliveries)} delivery records to {OUT_DIR}/deliveries.csv")
    print(f"Wrote {len(zones)} zone summaries to {OUT_DIR}/zones.csv")
