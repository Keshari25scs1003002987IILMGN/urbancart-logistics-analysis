"""
visualization.py

Week 3 task: Advanced Data Analysis and Visualization in Logistics.
Runs exploratory data analysis on UrbanCart's cleaned delivery dataset
and produces the visualizations referenced in the Week 3 Word report.

Usage:
    python generate_data.py
    python data_cleaning.py
    python visualization.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
OUT_DIR = "charts"
import os
os.makedirs(OUT_DIR, exist_ok=True)


def load_data(path="data/deliveries.csv"):
    df = pd.read_csv(path, parse_dates=["order_time", "promised_window_end", "actual_delivery"])
    df["on_time"] = df["actual_delivery"] <= df["promised_window_end"]
    df["delay_mins"] = ((df["actual_delivery"] - df["promised_window_end"]).dt.total_seconds() / 60).clip(lower=0)
    return df


def print_descriptive_stats(df: pd.DataFrame):
    print("=== Central Tendency & Spread ===")
    print(df[["distance_km", "delivery_time_hrs", "trip_cost", "hub_congestion"]].describe().round(2))

    print("\n=== Correlation Matrix ===")
    corr = df[["distance_km", "delivery_time_hrs", "trip_cost", "hub_congestion", "package_weight"]].corr()
    print(corr.round(2))
    return corr


def plot_delivery_time_distribution(df):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(df["delivery_time_hrs"], bins=30, kde=True, ax=ax, color="#4C72B0")
    ax.set_title("Distribution of Delivery Time (hours)")
    ax.set_xlabel("Delivery Time (hrs)")
    ax.set_ylabel("Number of Orders")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/01_delivery_time_distribution.png", dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(corr):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation Between Trip Features")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/02_correlation_heatmap.png", dpi=150)
    plt.close(fig)


def plot_cost_by_zone_boxplot(df):
    top_zones = df["zone_id"].value_counts().nlargest(10).index
    subset = df[df["zone_id"].isin(top_zones)]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=subset, x="zone_id", y="trip_cost", ax=ax, palette="Set2")
    ax.set_title("Trip Cost Distribution — Top 10 Zones by Order Volume")
    ax.set_xlabel("Zone ID")
    ax.set_ylabel("Trip Cost")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/03_cost_by_zone_boxplot.png", dpi=150)
    plt.close(fig)


def plot_otdr_by_zone_bar(df):
    zone_otdr = df.groupby("zone_id")["on_time"].mean().mul(100).sort_values().head(10)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    zone_otdr.plot(kind="barh", ax=ax, color="#DD8452")
    ax.set_title("10 Worst Zones by On-Time Delivery Rate")
    ax.set_xlabel("On-Time Delivery Rate (%)")
    ax.set_ylabel("Zone ID")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/04_otdr_worst_zones.png", dpi=150)
    plt.close(fig)


def plot_distance_vs_delivery_time_scatter(df):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.scatterplot(
        data=df.sample(min(500, len(df)), random_state=1),
        x="distance_km", y="delivery_time_hrs",
        hue="hub_congestion", palette="viridis", alpha=0.7, ax=ax,
    )
    ax.set_title("Distance vs. Delivery Time (colored by hub congestion)")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Delivery Time (hrs)")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/05_distance_vs_time_scatter.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    df = load_data()
    corr = print_descriptive_stats(df)

    plot_delivery_time_distribution(df)
    plot_correlation_heatmap(corr)
    plot_cost_by_zone_boxplot(df)
    plot_otdr_by_zone_bar(df)
    plot_distance_vs_delivery_time_scatter(df)

    print(f"\nSaved 5 charts to ./{OUT_DIR}/")
