"""
clustering.py

Segments UrbanCart's delivery zones into risk/cost groups using K-Means,
based on average delay, cost per order, and order density. The resulting
labels (low-risk, high-risk, high-volume) guide where to add riders or
renegotiate courier contracts.

Usage:
    python generate_data.py
    python clustering.py
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


FEATURES = ["avg_delay_mins", "cost_per_order", "order_density"]


def load_data(path="data/zones.csv"):
    return pd.read_csv(path)


def cluster_zones(df: pd.DataFrame, n_clusters=3):
    df = df.copy()
    scaled = StandardScaler().fit_transform(df[FEATURES])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(scaled)

    return df, kmeans


def label_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Rank clusters by average delay + cost to assign human-readable labels."""
    centroid_scores = df.groupby("cluster")[["avg_delay_mins", "cost_per_order"]].mean()
    centroid_scores["risk_score"] = centroid_scores["avg_delay_mins"] + centroid_scores["cost_per_order"]
    ranked = centroid_scores.sort_values("risk_score").index.tolist()

    label_map = {}
    labels = ["low-risk / low-cost", "medium-risk", "high-risk / high-cost"]
    for i, cluster_id in enumerate(ranked):
        label_map[cluster_id] = labels[min(i, len(labels) - 1)]

    df = df.copy()
    df["cluster_label"] = df["cluster"].map(label_map)
    return df


if __name__ == "__main__":
    df = load_data()
    clustered, model = cluster_zones(df)
    labeled = label_clusters(clustered)

    print("=== Zone Clusters ===")
    print(labeled.sort_values("cluster").to_string(index=False))

    print("\n=== Cluster Summary ===")
    print(labeled.groupby("cluster_label")[FEATURES].mean().round(2))
