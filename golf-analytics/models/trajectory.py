"""
Trajectory engine — is this player's game moving, and can you trust it?

Per player (on differentials — the absolute scale, so improvement means
improvement, not softer fields):

1. RECENT SLOPE: OLS on the last 30 rounds -> strokes per 10 rounds, with
   a standard error so noise can't masquerade as a trend.
2. EARLIER SLOPE: same fit on the rounds before that window, so we can
   tell "still climbing" from "climbed, then flattened" (plateau).
3. FLUKE SCAN: slide an 18-round window across the season; if some stretch
   is >= 2 strokes better than BOTH what came before and what came after,
   the hot streak reverted — flag it so a ranking built on that stretch
   gets read with caution.

Classifications: Rapidly Improving (watchlist) / Improving / Steady /
Plateauing / Declining, plus an independent fluke_flag.

Usage:  python models/trajectory.py
Writes: player_trajectory table in DB
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from config import DATABASE_URL

MIN_ROUNDS = 15
RECENT_WINDOW = 30
FLUKE_WINDOW = 12
FLUKE_GAP = 2.2        # strokes better than both neighbors
RAPID_SLOPE = -0.45    # strokes per 10 rounds
IMPROVE_SLOPE = -0.18
DECLINE_SLOPE = 0.18
Z_TREND = 1.6
Z_RAPID = 2.2       # stricter: watchlist must be statistically solid


def ols_slope(y: np.ndarray):
    """slope per round + its SE."""
    x = np.arange(len(y), dtype=float)
    n = len(y)
    if n < 8:
        return None, None
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    se = np.sqrt(resid.var(ddof=2) / ((x - x.mean()) ** 2).sum())
    return slope, se


def fluke_scan(y: np.ndarray):
    """Best hot streak that reverted: returns (start, gap) or None."""
    n = len(y)
    if n < FLUKE_WINDOW + 10:
        return None
    best = None
    for s in range(5, n - FLUKE_WINDOW - 5):
        block = y[s:s + FLUKE_WINDOW].mean()
        before = y[:s].mean()
        after = y[s + FLUKE_WINDOW:].mean()
        gap = min(before - block, after - block)   # positive = block better than both
        se = y.std(ddof=1) / np.sqrt(FLUKE_WINDOW)
        if gap >= FLUKE_GAP and gap / se >= 2.5 and (best is None or gap > best[1]):
            best = (s, round(gap, 2))
    return best


def classify(slope10, z, earlier_slope10, earlier_z):
    if slope10 is None:
        return "Insufficient data"
    if slope10 <= RAPID_SLOPE and z <= -Z_RAPID:
        return "Rapidly Improving"
    if slope10 <= IMPROVE_SLOPE and z <= -Z_TREND:
        return "Improving"
    if slope10 >= DECLINE_SLOPE and z >= Z_TREND:
        return "Declining"
    # flat now — was it climbing before? that's a plateau, not steady
    if (earlier_slope10 is not None and earlier_slope10 <= IMPROVE_SLOPE
            and earlier_z is not None and earlier_z <= -Z_TREND):
        return "Plateauing"
    return "Steady"


def main():
    engine = create_engine(DATABASE_URL)
    df = pd.read_sql("""
        SELECT player_id, round_date, round_num, differential
        FROM mart_player_rounds
    """, engine).sort_values(["player_id", "round_date", "round_num"])

    rows = []
    for pid, g in df.groupby("player_id"):
        y = g.differential.to_numpy()
        if len(y) < MIN_ROUNDS:
            continue

        recent = y[-RECENT_WINDOW:]
        earlier = y[:-RECENT_WINDOW] if len(y) > RECENT_WINDOW + 8 else np.array([])

        slope, se = ols_slope(recent)
        slope10 = slope * 10 if slope is not None else None
        z = slope / se if (slope is not None and se and se > 0) else 0.0

        e_slope, e_se = (ols_slope(earlier) if len(earlier) >= 10 else (None, None))
        e_slope10 = e_slope * 10 if e_slope is not None else None
        e_z = e_slope / e_se if (e_slope is not None and e_se and e_se > 0) else None

        fluke = fluke_scan(y)

        label = classify(slope10, z, e_slope10, e_z)
        rows.append(dict(
            player_id=pid,
            rounds=len(y),
            recent_slope_per10=round(slope10, 3) if slope10 is not None else None,
            slope_z=round(z, 2),
            earlier_slope_per10=round(e_slope10, 3) if e_slope10 is not None else None,
            season_diff=round(y.mean(), 2),
            recent_diff=round(recent.mean(), 2),
            trajectory=label,
            watchlist=bool(label == "Rapidly Improving"),
            fluke_flag=bool(fluke is not None),
            fluke_gap=fluke[1] if fluke else None,
        ))

    traj = pd.DataFrame(rows)
    names = pd.read_sql(
        "SELECT player_id, player_name, level, team, class_year FROM stg_players",
        engine)
    traj = traj.merge(names, on="player_id")
    traj.to_sql("player_trajectory", engine, if_exists="replace", index=False)

    print(traj.groupby(["level", "trajectory"]).size().to_string())
    print(f"\nwatchlist (rapid improvers): {int(traj.watchlist.sum())}")
    print(traj[traj.watchlist][["player_name", "level", "team", "class_year",
                                "recent_slope_per10", "slope_z", "recent_diff"]]
          .sort_values("recent_slope_per10").head(12).to_string(index=False))
    print(f"\nfluke-season flags: {int(traj.fluke_flag.sum())}")
    print(traj[traj.fluke_flag][["player_name", "level", "fluke_gap",
                                 "trajectory"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
