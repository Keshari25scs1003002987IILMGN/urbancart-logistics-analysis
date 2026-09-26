"""
route_optimization.py

Assigns delivery stops to a fleet of vehicles and sequences each
vehicle's route using a capacity-constrained nearest-neighbor heuristic.
This is a lightweight stand-in for a full Vehicle Routing Problem (VRP)
solver (e.g. Google OR-Tools) -- same objective (minimize total travel
distance under vehicle capacity), simpler implementation so it runs
with no extra dependencies.

Usage:
    python generate_data.py
    python route_optimization.py
"""

import numpy as np
import pandas as pd

np.random.seed(7)

N_STOPS = 25
N_VEHICLES = 4
VEHICLE_CAPACITY = 8  # max stops per vehicle
HUB_COORD = (0.0, 0.0)


def generate_stops(n=N_STOPS):
    """Synthetic delivery-stop coordinates (km offsets from the hub)."""
    coords = np.random.uniform(-15, 15, size=(n, 2))
    return pd.DataFrame({
        "stop_id": np.arange(1, n + 1),
        "x": coords[:, 0],
        "y": coords[:, 1],
    })


def distance(a, b):
    return np.hypot(a[0] - b[0], a[1] - b[1])


def nearest_neighbor_routes(stops: pd.DataFrame, n_vehicles=N_VEHICLES, capacity=VEHICLE_CAPACITY):
    """
    Greedily assigns the nearest unvisited stop to whichever vehicle
    is currently closest to it, subject to a per-vehicle capacity cap.
    """
    remaining = stops.copy()
    vehicle_routes = {v: [] for v in range(1, n_vehicles + 1)}
    vehicle_pos = {v: HUB_COORD for v in range(1, n_vehicles + 1)}
    vehicle_load = {v: 0 for v in range(1, n_vehicles + 1)}

    while not remaining.empty:
        best = None  # (vehicle, stop_index, dist)
        for v in vehicle_routes:
            if vehicle_load[v] >= capacity:
                continue
            dists = remaining.apply(lambda r: distance(vehicle_pos[v], (r["x"], r["y"])), axis=1)
            idx = dists.idxmin()
            d = dists[idx]
            if best is None or d < best[2]:
                best = (v, idx, d)

        if best is None:
            # all vehicles at capacity but stops remain -> would need more vehicles
            break

        v, idx, d = best
        stop = remaining.loc[idx]
        vehicle_routes[v].append(int(stop["stop_id"]))
        vehicle_pos[v] = (stop["x"], stop["y"])
        vehicle_load[v] += 1
        remaining = remaining.drop(idx)

    return vehicle_routes, remaining


def route_distance(stops: pd.DataFrame, route_ids):
    pos = HUB_COORD
    total = 0.0
    for sid in route_ids:
        row = stops[stops["stop_id"] == sid].iloc[0]
        total += distance(pos, (row["x"], row["y"]))
        pos = (row["x"], row["y"])
    total += distance(pos, HUB_COORD)  # return to hub
    return round(total, 2)


if __name__ == "__main__":
    stops = generate_stops()
    routes, unassigned = nearest_neighbor_routes(stops)

    print(f"Optimizing {len(stops)} stops across {N_VEHICLES} vehicles "
          f"(capacity {VEHICLE_CAPACITY} stops/vehicle)\n")

    total_distance = 0.0
    for v, route in routes.items():
        d = route_distance(stops, route)
        total_distance += d
        print(f"Vehicle {v}: {len(route)} stops, route distance = {d} km -> {route}")

    if not unassigned.empty:
        print(f"\nUnassigned stops (exceeds total fleet capacity): "
              f"{list(unassigned['stop_id'])}")

    print(f"\nTotal fleet distance: {round(total_distance, 2)} km")
