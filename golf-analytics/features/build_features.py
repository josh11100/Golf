"""
Build the ML feature table from the SQL marts.

Target: adj_score for the round (score relative to field — course/condition
normalized, so the model learns player quality + context, not course quirks).

Features are strictly PRE-ROUND information (no leakage):
- rolling_adj_5      last-5-rounds form (excludes current round, built in SQL)
- prev_adj           previous round adjusted score
- career_round_num   experience
- season std dev     consistency up to (not including) this round
- field_strength     tournament field quality
- round_num          R1/R2/R3 within tournament
- weather            temp / wind / precip forecastable pre-round
- class year         FR/SO/JR/SR encoded

Usage:  python features/build_features.py
Writes: feature_table in the database
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from config import DATABASE_URL

CLASS_MAP = {"FR": 1, "SO": 2, "JR": 3, "SR": 4}


def main():
    engine = create_engine(DATABASE_URL)

    rounds = pd.read_sql("SELECT * FROM mart_player_rounds", engine)
    players = pd.read_sql("SELECT player_id, class_year FROM stg_players", engine)
    field = pd.read_sql(
        "SELECT tournament_id, field_strength FROM mart_field_strength", engine)

    df = (rounds
          .merge(players, on="player_id", how="left")
          .merge(field, on="tournament_id", how="left"))

    df = df.sort_values(["player_id", "round_date", "round_num"])

    # expanding (career-to-date) consistency, shifted so current round excluded
    df["expanding_std"] = (df.groupby("player_id")["adj_score"]
                             .transform(lambda s: s.shift(1).expanding(min_periods=3).std()))

    df["class_num"] = df["class_year"].map(CLASS_MAP).fillna(2)

    feature_cols = [
        "rolling_adj_5", "prev_adj", "career_round_num", "expanding_std",
        "field_strength", "event_tier", "round_num", "temp_f", "wind_mph", "precip_in",
        "class_num",
    ]
    keep = ["round_id", "player_id", "tournament_id", "round_date",
            "adj_score"] + feature_cols

    out = df[keep].dropna(subset=["rolling_adj_5", "prev_adj", "expanding_std"])
    out.to_sql("feature_table", engine, if_exists="replace", index=False)

    print(f"feature_table: {len(out):,} rows, {len(feature_cols)} features")
    print(f"(dropped {len(df) - len(out):,} early-career rounds without form history)")


if __name__ == "__main__":
    main()
