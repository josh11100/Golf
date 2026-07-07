"""
XGBoost round-performance prediction.

Predicts a player's adjusted score (vs field) for an upcoming round from
pre-round features. Time-based split (train on early season, test on late
season) — a random split would leak future form into the past.

Usage:  python models/train_performance.py
Writes: models/artifacts/xgb_performance.joblib
        models/artifacts/xgb_metrics.json
        predictions table in DB (test-set predictions for the dashboard)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sqlalchemy import create_engine
from xgboost import XGBRegressor

from config import ARTIFACTS_DIR, DATABASE_URL

FEATURES = [
    "rolling_adj_5", "prev_adj", "career_round_num", "expanding_std",
    "field_strength", "event_tier", "round_num", "temp_f", "wind_mph", "precip_in",
    "class_num",
]
TARGET = "adj_score"


def main():
    engine = create_engine(DATABASE_URL)
    df = pd.read_sql("SELECT * FROM feature_table", engine)
    df["round_date"] = pd.to_datetime(df["round_date"])
    df = df.sort_values("round_date")

    # time-based 80/20 split
    cutoff = df["round_date"].quantile(0.8)
    train, test = df[df.round_date <= cutoff], df[df.round_date > cutoff]

    # heavily regularized: golf rounds are mostly noise, so a shallow,
    # slow-learning model generalizes better than a deep one
    model = XGBRegressor(
        n_estimators=500,
        max_depth=2,
        learning_rate=0.02,
        min_child_weight=40,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=20.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train[FEATURES], train[TARGET])

    pred = model.predict(test[FEATURES])
    mae = mean_absolute_error(test[TARGET], pred)
    r2 = r2_score(test[TARGET], pred)

    # baseline: "tomorrow looks like your rolling form" — the bar to beat
    base_mae = mean_absolute_error(test[TARGET], test["rolling_adj_5"])

    print(f"train {len(train):,} | test {len(test):,} (cutoff {cutoff.date()})")
    print(f"XGBoost   MAE {mae:.3f} strokes | R^2 {r2:.3f}")
    print(f"Baseline  MAE {base_mae:.3f} strokes (rolling-form naive)")

    importances = dict(zip(FEATURES, model.feature_importances_.round(4).tolist()))
    print("feature importance:",
          json.dumps(dict(sorted(importances.items(), key=lambda x: -x[1])), indent=2))

    joblib.dump(model, ARTIFACTS_DIR / "xgb_performance.joblib")
    (ARTIFACTS_DIR / "xgb_metrics.json").write_text(json.dumps({
        "mae": round(float(mae), 4),
        "r2": round(float(r2), 4),
        "baseline_mae": round(float(base_mae), 4),
        "n_train": len(train),
        "n_test": len(test),
        "cutoff": str(cutoff.date()),
        "feature_importance": importances,
    }, indent=2))

    out = test[["round_id", "player_id", "tournament_id", "round_date", TARGET]].copy()
    out["predicted_adj"] = pred
    out.to_sql("predictions", engine, if_exists="replace", index=False)
    print(f"saved model + metrics -> {ARTIFACTS_DIR}, predictions -> DB")


if __name__ == "__main__":
    main()
