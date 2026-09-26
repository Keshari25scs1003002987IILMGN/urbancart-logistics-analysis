# UrbanCart Logistics Analysis

Week 1 task — Logistics Data Analyst Internship (YuvaIntern).

Simulates a logistics data pipeline for **UrbanCart**, a mid-sized
e-commerce retailer facing delivery delays and rising last-mile costs.
The scripts here implement the KPIs, roadmap, and Python approach
described in the accompanying Word report.

## What's here

| File | Purpose |
|---|---|
| `generate_data.py` | Creates synthetic `data/deliveries.csv` and `data/zones.csv` in place of real order-management / fleet-tracking exports |
| `kpi_calculation.py` | Computes On-Time Delivery Rate, Average Delivery Cost per Order, and Order Fulfillment Cycle Time (overall and by zone) |
| `regression_model.py` | Random Forest model that predicts delivery time from distance, hub congestion, package weight, and hour of day |
| `clustering.py` | K-Means clustering that segments delivery zones into low/medium/high-risk groups |
| `route_optimization.py` | Capacity-constrained nearest-neighbor heuristic that assigns and sequences delivery stops per vehicle (a lightweight stand-in for a full VRP solver such as Google OR-Tools) |

## Setup

```bash
pip install -r requirements.txt
```

## Run order

```bash
python generate_data.py       # 1. create the datasets
python kpi_calculation.py     # 2. overall + per-zone KPIs
python regression_model.py    # 3. delivery-time prediction model
python clustering.py          # 4. zone risk/cost segmentation
python route_optimization.py  # 5. route assignment + distance summary
```

## Notes

- All data is synthetically generated (`np.random.seed` fixed for
  reproducibility) since real UrbanCart order data isn't available for
  this task — the pipeline structure and modeling approach are what
  the internship task asks to demonstrate.
- `route_optimization.py` uses a simple nearest-neighbor heuristic
  rather than an external solver so the project has no dependency
  beyond `requirements.txt`. The objective (minimize total travel
  distance under vehicle capacity) mirrors what a full VRP solver
  optimizes.
