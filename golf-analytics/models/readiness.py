"""
College readiness — where does a junior stand against actual D1 golf?

Both populations are scored on the same absolute scale (USGA differentials:
(score - rating) * 113 / slope), so a junior's recent-20 differential can be
placed directly on the distribution of current college players' recent-20
differentials. Percentile -> a plain-language band a coach and a family can
both read.

Usage:  python models/readiness.py
Writes: player_readiness table in DB (juniors only)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from config import DATABASE_URL

MIN_ROUNDS = 12

BANDS = [
    (90, "High-major starter range"),
    (70, "Mid-major starter / high-major depth"),
    (50, "Mid-major lineup range"),
    (30, "Low-major / developmental range"),
    (0,  "Below current college range"),
]


def band(pct: float) -> str:
    for cut, label in BANDS:
        if pct >= cut:
            return label
    return BANDS[-1][1]


def main():
    engine = create_engine(DATABASE_URL)
    s = pd.read_sql("""
        SELECT player_id, player_name, level, team, class_year,
               rounds_played, recent_diff_20
        FROM mart_player_summary
    """, engine)

    college = s[(s.level == "college") & (s.rounds_played >= MIN_ROUNDS)]
    juniors = s[(s.level == "junior") & (s.rounds_played >= MIN_ROUNDS)].copy()
    dist = np.sort(college.recent_diff_20.to_numpy())

    # percentile = share of college players this junior currently outscores
    juniors["college_percentile"] = juniors.recent_diff_20.apply(
        lambda d: round(100.0 * (dist > d).mean(), 1))
    juniors["readiness_band"] = juniors.college_percentile.apply(band)
    # nearest comparable college player (same absolute scale)
    def nearest(d):
        i = (college.recent_diff_20 - d).abs().idxmin()
        r = college.loc[i]
        return f"{r.player_name} ({r.team})"
    juniors["comparable_college_player"] = juniors.recent_diff_20.apply(nearest)

    out = juniors[["player_id", "player_name", "team", "class_year",
                   "rounds_played", "recent_diff_20", "college_percentile",
                   "readiness_band", "comparable_college_player"]]
    out.to_sql("player_readiness", engine, if_exists="replace", index=False)

    print(f"juniors scored: {len(out)} (vs {len(college)} college players)")
    print(out.readiness_band.value_counts().to_string())
    print("\ntop 10 college-ready juniors:")
    print(out.sort_values("college_percentile", ascending=False)
          [["player_name", "class_year", "recent_diff_20",
            "college_percentile", "readiness_band"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
