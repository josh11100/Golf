"""
Performance-drop anomaly detection (changepoint scan + IsolationForest).

For each player, scan every candidate split point in their season and find
the one that maximizes the standardized jump in adjusted scoring between
"before" and "after" (a simple mean-shift changepoint detector). A player
is flagged when:
  1. the detected shift is a sustained WORSENING (>= MIN_DROP strokes),
  2. it is statistically large (z >= Z_THRESHOLD), and
  3. IsolationForest agrees they're an outlier across the population.

This catches mid-season drops (injury, swing change, burnout) that a
fixed last-N-rounds window dilutes.

Usage:  python models/anomaly_detection.py
Writes: anomaly_flags table in DB
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sqlalchemy import create_engine

from config import DATABASE_URL

MIN_ROUNDS = 18       # need enough season to detect a shift
MIN_SEGMENT = 6       # min rounds on each side of the changepoint
MIN_DROP = 1.5        # strokes worse, sustained
Z_THRESHOLD = 2.0


def best_changepoint(scores: np.ndarray):
    """Return (split_idx, drop, z) for the strongest mean-shift worsening."""
    n = len(scores)
    best = (None, 0.0, 0.0)
    for s in range(MIN_SEGMENT, n - MIN_SEGMENT + 1):
        before, after = scores[:s], scores[s:]
        drop = after.mean() - before.mean()          # positive = worse
        pooled = np.sqrt(before.var(ddof=1) / len(before)
                         + after.var(ddof=1) / len(after))
        if pooled == 0 or np.isnan(pooled):
            continue
        z = drop / pooled
        if z > best[2]:
            best = (s, drop, z)
    return best


def main():
    engine = create_engine(DATABASE_URL)
    df = pd.read_sql(
        "SELECT player_id, round_date, round_num, adj_score FROM mart_player_rounds",
        engine).sort_values(["player_id", "round_date", "round_num"])

    rows = []
    for pid, g in df.groupby("player_id"):
        if len(g) < MIN_ROUNDS:
            continue
        scores = g.adj_score.to_numpy()
        split, drop, z = best_changepoint(scores)
        if split is None:
            continue
        rows.append(dict(
            player_id=pid,
            rounds=len(g),
            changepoint_round=split,
            baseline_adj=round(scores[:split].mean(), 2),
            recent_adj=round(scores[split:].mean(), 2),
            drop_strokes=round(drop, 2),
            z_score=round(z, 2),
        ))

    flags = pd.DataFrame(rows)

    iso = IsolationForest(contamination=0.08, random_state=42)
    flags["iso_outlier"] = (iso.fit_predict(
        flags[["drop_strokes", "z_score"]].values) == -1)

    flags["flagged"] = ((flags.drop_strokes >= MIN_DROP)
                        & (flags.z_score >= Z_THRESHOLD)
                        & flags.iso_outlier)
    flags = flags.sort_values("z_score", ascending=False)

    names = pd.read_sql("SELECT player_id, player_name, team FROM stg_players",
                        engine)
    flags = flags.merge(names, on="player_id")

    n = int(flags.flagged.sum())
    print(f"{n} players flagged (sustained drop >= {MIN_DROP} strokes, "
          f"z >= {Z_THRESHOLD}, isolation outlier)")
    print(flags[flags.flagged][["player_name", "team", "changepoint_round",
                                "baseline_adj", "recent_adj",
                                "drop_strokes", "z_score"]]
          .head(15).to_string(index=False))

    flags.to_sql("anomaly_flags", engine, if_exists="replace", index=False)
    print("saved anomaly_flags table -> DB")


if __name__ == "__main__":
    main()
