"""
Player archetype clustering (KMeans).

Clusters players on season profile: scoring level, volatility, top-finish
rate, and trend. Produces labeled archetypes like "Elite Grinder" vs
"Boom/Bust" — the season-long personality of a player, not one result.

Usage:  python models/player_archetypes.py
Writes: player_archetypes table in DB, kmeans model artifact
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine

from config import ARTIFACTS_DIR, DATABASE_URL

FEATURES = ["avg_adj_score", "adj_score_std", "top10pct_rate", "season_trend"]
K_RANGE = range(3, 8)
MIN_ROUNDS = 8


def label_clusters(centers: pd.DataFrame) -> dict:
    """Human-readable archetype names from cluster centers (original units)."""
    labels = {}
    for i, c in centers.iterrows():
        good = c.avg_adj_score < centers.avg_adj_score.median()
        steady = c.adj_score_std < centers.adj_score_std.median()
        improving = c.season_trend < -0.15
        if good and steady:
            name = "Elite Grinder"
        elif good and not steady:
            name = "High-Ceiling Volatile"
        elif not good and steady:
            name = "Steady Mid-Pack"
        else:
            name = "Boom/Bust"
        if improving:
            name += " (Rising)"
        labels[i] = name
    return labels


def main():
    engine = create_engine(DATABASE_URL)
    df = pd.read_sql("SELECT * FROM mart_player_summary", engine)
    df = df[df.rounds_played >= MIN_ROUNDS].dropna(subset=FEATURES).copy()

    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES])

    # pick k by silhouette
    best_k, best_s = None, -1
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        s = silhouette_score(X, km.labels_)
        print(f"k={k}: silhouette {s:.3f}")
        if s > best_s:
            best_k, best_s = k, s

    km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X)
    df["cluster"] = km.labels_

    centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_),
                           columns=FEATURES)
    names = label_clusters(centers)
    df["archetype"] = df["cluster"].map(names)

    print(f"\nchose k={best_k} (silhouette {best_s:.3f})")
    print(df.groupby("archetype")[FEATURES + ["rounds_played"]]
            .mean().round(2).to_string())

    df.to_sql("player_archetypes", engine, if_exists="replace", index=False)
    joblib.dump({"kmeans": km, "scaler": scaler, "features": FEATURES,
                 "names": names}, ARTIFACTS_DIR / "kmeans_archetypes.joblib")
    print(f"\nsaved player_archetypes table + artifact -> {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
