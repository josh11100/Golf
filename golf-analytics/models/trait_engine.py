"""
Player trait engine — "how does this player's game behave?"

Traits are estimated as performance differences across contexts, then
SHRUNK toward zero (empirical-Bayes style) so small samples can't earn
badges by luck. A trait only surfaces when the sample is big enough AND
the effect survives shrinkage AND a Welch z-test agrees. Silence is a
feature: "insufficient data" beats a fake badge a coach will catch.

Traits:
  big_game      differential in high-tier events (tier >= 3.5) vs lower —
                differential is absolute, so course difficulty is already
                priced in and stronger fields don't mechanically punish you
  closer        adj_score in final round vs earlier rounds (same-day field
                comparison handles conditions)
  fast_starter  adj_score in round 1 vs later rounds
  wind_player   slope of adj_score vs wind (per +10 mph, field-relative so
                only *relative* wind skill remains)

Usage:  python models/trait_engine.py
Writes: player_traits table in DB
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from config import DATABASE_URL

MIN_BUCKET = 8        # min rounds on each side of a context split
SHRINK_K = 15         # rounds of "prior" pulling every effect toward zero
BADGE_EFFECT = 0.6    # strokes, post-shrinkage
BADGE_Z = 1.8

TIER_SPLIT = 3.5
WIND_SPLIT = 15.0     # mph, for sample-size checks on the wind trait


def welch(a: np.ndarray, b: np.ndarray):
    """effect = mean(b) - mean(a) with Welch z. Positive = better in a."""
    ma, mb = a.mean(), b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    if se == 0 or np.isnan(se):
        return 0.0, 0.0
    return mb - ma, (mb - ma) / se


def shrink(effect: float, n_eff: float) -> float:
    return effect * n_eff / (n_eff + SHRINK_K)


def eval_split(hi: pd.Series, lo: pd.Series):
    """Generic two-bucket trait: positive effect = better in `hi` bucket."""
    if len(hi) < MIN_BUCKET or len(lo) < MIN_BUCKET:
        return None
    effect, z = welch(hi.to_numpy(), lo.to_numpy())   # mean(lo) - mean(hi)
    n_eff = 2 * len(hi) * len(lo) / (len(hi) + len(lo))
    return dict(raw_effect=round(effect, 2),
                shrunk_effect=round(shrink(effect, n_eff), 2),
                z=round(z, 2), n_high=len(hi), n_low=len(lo))


def wind_trait(g: pd.DataFrame):
    w = g.dropna(subset=["wind_mph"])
    calm, breezy = w[w.wind_mph < WIND_SPLIT], w[w.wind_mph >= WIND_SPLIT]
    if len(calm) < MIN_BUCKET or len(breezy) < MIN_BUCKET:
        return None
    slope = np.polyfit(w.wind_mph, w.adj_score, 1)[0] * 10   # per +10 mph
    # effect sign convention: positive = handles wind BETTER than field
    effect = -slope
    resid = w.adj_score - np.poly1d(np.polyfit(w.wind_mph, w.adj_score, 1))(w.wind_mph)
    se = resid.std(ddof=2) / (w.wind_mph.std() * np.sqrt(len(w))) * 10
    z = effect / se if se > 0 else 0.0
    n_eff = 2 * len(calm) * len(breezy) / (len(calm) + len(breezy))
    return dict(raw_effect=round(effect, 2),
                shrunk_effect=round(shrink(effect, n_eff), 2),
                z=round(z, 2), n_high=len(breezy), n_low=len(calm))


DESCRIPTIONS = {
    ("big_game", 1): "Plays UP in stronger fields",
    ("big_game", -1): "Fades in stronger fields",
    ("closer", 1): "Closes — final rounds beat their baseline",
    ("closer", -1): "Final-round fader",
    ("fast_starter", 1): "Fast starter — hot in round 1",
    ("fast_starter", -1): "Slow starter",
    ("wind_player", 1): "Wind player — loses less than the field in breeze",
    ("wind_player", -1): "Wind-sensitive",
}


def main():
    engine = create_engine(DATABASE_URL)
    df = pd.read_sql("""
        SELECT player_id, adj_score, differential, event_tier, round_num,
               num_rounds, wind_mph
        FROM mart_player_rounds
    """, engine)

    rows = []
    for pid, g in df.groupby("player_id"):
        results = {
            # differential (absolute scale): are you better in big events?
            "big_game": eval_split(
                g[g.event_tier >= TIER_SPLIT].differential,
                g[g.event_tier < TIER_SPLIT].differential),
            # adj_score (field-relative): do you beat the field when it matters?
            "closer": eval_split(
                g[(g.round_num == g.num_rounds) & (g.num_rounds > 1)].adj_score,
                g[g.round_num < g.num_rounds].adj_score),
            "fast_starter": eval_split(
                g[g.round_num == 1].adj_score,
                g[g.round_num > 1].adj_score),
            "wind_player": wind_trait(g),
        }
        for trait, r in results.items():
            if r is None:
                continue
            badge = abs(r["shrunk_effect"]) >= BADGE_EFFECT and abs(r["z"]) >= BADGE_Z
            direction = 1 if r["shrunk_effect"] > 0 else -1
            rows.append(dict(
                player_id=pid, trait=trait, **r,
                badge=bool(badge),
                description=DESCRIPTIONS[(trait, direction)] if badge else None,
            ))

    traits = pd.DataFrame(rows)
    names = pd.read_sql("SELECT player_id, player_name, level, team FROM stg_players",
                        engine)
    traits = traits.merge(names, on="player_id")
    traits.to_sql("player_traits", engine, if_exists="replace", index=False)

    badged = traits[traits.badge]
    print(f"trait rows: {len(traits):,} | badges earned: {len(badged):,} "
          f"across {badged.player_id.nunique()} players")
    print(badged.groupby(["trait"]).size().to_string())
    print("\nsample badges:")
    print(badged.sort_values("z", key=abs, ascending=False)
          [["player_name", "level", "trait", "shrunk_effect", "z",
            "n_high", "n_low", "description"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
