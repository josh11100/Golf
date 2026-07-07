# ⛳ Golf Performance Intelligence — College + Junior Recruiting

One platform, two populations, one scale. College (Clippd Scoreboard, the official NCAA source) and junior
golf (AJGA, Junior Golf Scoreboard, FCG, Toyota Tour Cup, SDJGA, HSJGA,
NCJGA) are scored on the **same absolute scale — USGA differentials**
`(score − course rating) × 113 / slope` — so a junior's game can be placed
directly on the current D1 distribution. That's the recruiting product:
*"your last 20 rounds put you at the 74th percentile of current D1 players."*

**Two comparison scales, used deliberately:**
- **Differential** (absolute) — bridges junior ↔ college, powers readiness
  percentiles and improvement tracking (gains are real, not softer fields)
- **Field-adjusted score** (relative) — score vs that day's field average;
  powers within-level leaderboards and same-day traits like closing

## Stack

| Layer | Tech |
|---|---|
| Ingestion | Async JSON ingestion targeting Clippd Scoreboard (official NCAA source — Golfstat retired) + junior tour parsers |
| Storage / SQL | PostgreSQL (SQLite for local dev), marts built with CTEs + window functions |
| Weather | Open-Meteo (historical backfill), OpenWeather (current/forecast) |
| ML | XGBoost (round prediction), KMeans (archetypes), IsolationForest + z-scores (anomaly detection) |
| App | Streamlit + Plotly |

## Quick start (runs in ~2 minutes)

```bash
pip install -r requirements.txt
cp .env.example .env          # add your OpenWeather key

# 1. Data — synthetic season until scraper is tuned (same schema)
python data/generate_sample_data.py

# 2. Load staging + build SQL marts
python etl/load.py

# 3. Feature engineering
python features/build_features.py

# 4. Train models + engines
python models/train_performance.py     # XGBoost round prediction
python models/player_archetypes.py     # KMeans archetypes
python models/anomaly_detection.py     # changepoint collapse detection
python models/trait_engine.py          # shrinkage-tested strength badges
python models/trajectory.py            # improvement / plateau / fluke detection
python models/readiness.py             # junior -> D1 percentile

# 5. Dashboard
streamlit run app/dashboard.py
```

Default DB is SQLite (`golf.db`) so it runs instantly. For the real build,
set in `.env`:

```
DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/golf
```

All SQL is portable across both.

## Project layout

```
├── config.py                     # env config (DB URL, API keys)
├── db/schema.sql                 # staging tables
├── sql/marts.sql                 # marts: CTEs + window functions (the SQL showcase)
├── scraper/college_scraper.py    # Clippd Scoreboard JSON ingestion (endpoints via DevTools — see docstring)
├── data/generate_sample_data.py  # realistic synthetic season, same schema as scraper output
├── etl/load.py                   # CSV -> staging -> marts
├── features/
│   ├── weather.py                # Open-Meteo historical backfill + OpenWeather current
│   └── build_features.py         # leakage-safe pre-round feature table
├── models/
│   ├── train_performance.py      # XGBoost, time-based split, naive-form baseline
│   ├── player_archetypes.py      # KMeans, k by silhouette, named archetypes
│   └── anomaly_detection.py      # rolling z-score + IsolationForest consensus
└── app/dashboard.py              # Streamlit dashboard
```

## The SQL marts (sql/marts.sql)

- **`mart_player_rounds`** — per-round enrichment: field average, field rank,
  adjusted score, 5-round rolling form (`ROWS BETWEEN 5 PRECEDING AND 1
  PRECEDING` — excludes the current round so it's a legal pre-round feature),
  `LAG` momentum, career round numbering.
- **`mart_field_strength`** — tournament field quality from entrants'
  season-to-date adjusted averages.
- **`mart_player_summary`** — season profile per player: level, volatility,
  top-10% rate, early/late-season trend split.
- **`mart_team_leaderboard`** — team standings on adjusted scoring.

## Recruiting intelligence layer

- **College readiness (`models/readiness.py`)** — junior's recent-20
  differential placed on the distribution of current college players' recent
  form -> percentile, plain-language band, and nearest college comp.
- **Trait engine (`models/trait_engine.py`)** — "how does their game
  behave?" Big-game (up/down vs event tier, built from AJGA-points-style
  tiers), closer, fast starter, wind player. Every effect is shrunk toward
  zero (empirical Bayes) and Welch-tested, with minimum-sample gates — small
  samples cannot buy a badge, and "insufficient data" is a valid output.
- **Trajectory engine (`models/trajectory.py`)** — OLS trend on recent
  differentials with standard errors: Rapidly Improving (coach watchlist),
  Improving, Steady, Plateauing (climbed, then flattened), Declining. Plus a
  **fluke scan**: a sliding-window search for hot streaks that reverted, so
  a ranking built on one hot month gets read with caution.

## The models

- **Round prediction (XGBoost)** — predicts adjusted score from strictly
  pre-round features (form, momentum, consistency, field strength, weather,
  experience). Time-based train/test split; reported against a naive
  "you'll shoot your rolling form" baseline so the lift is honest.
- **Archetypes (KMeans)** — clusters season profiles into named archetypes
  (Elite Grinder, High-Ceiling Volatile, Steady Mid-Pack, Boom/Bust, with a
  Rising modifier), k selected by silhouette score.
- **Anomaly watch** — flags sustained performance drops only when a player's
  recent form is ≥1.5σ off their own baseline **and** an IsolationForest
  marks them as a population outlier. Built as an early "something's off"
  signal (injury, swing change, burnout).

## Swapping in real data

`data/generate_sample_data.py` and the scraper produce identical CSV
schemas. Tune the scraper's selectors (see its docstring for the exact
workflow), run it, run `python features/weather.py backfill` to attach real
historical weather, then re-run the pipeline from step 2. Nothing else
changes.

## Notes

- `.env` is gitignored — never commit API keys. Rotate any key that has been
  shared in plaintext.
- Scrape responsibly: low concurrency, delays, honest User-Agent, check
  robots.txt/ToS.
