"""
Generate a realistic synthetic college + junior golf dataset.

Same contract as before: identical schema to scraper output, so tuning the
scrapers later swaps real data in without touching anything downstream.

Realism baked in (this is what the models are built to recover):
- Latent skill + consistency profiles (grinders vs boom/bust)
- Course difficulty + USGA course rating/slope on every event
  -> differentials bridge juniors and college onto one absolute scale
- Unified event_tier (1-5) from tour prestige (AJGA invitational != SDJGA
  local), so "competition level" exists as signal
- TRAITS: some players genuinely play up in big events (big-game), close
  better in final rounds (closer), start hot (fast starter), or handle
  wind better/worse (per-player wind sensitivity)
- TRAJECTORIES (juniors especially): rapid improvers, steady improvers,
  plateauers, decliners, and "fluke season" players who spike for a
  stretch and revert
- A few injury-style mid-season collapses (anomaly detection target)

Usage:  python data/generate_sample_data.py
Output: data/raw/players.csv, tournaments.csv, rounds.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import RAW_DIR

rng = np.random.default_rng(42)

SEASON_START = pd.Timestamp("2025-09-08")

# ----------------------------------------------------------- college config
TEAMS = [
    "UC Santa Barbara", "Pepperdine", "San Diego State", "USC", "UCLA",
    "Stanford", "Cal", "Arizona State", "Arizona", "Oregon",
    "Washington", "Texas", "Oklahoma State", "Vanderbilt", "Florida",
    "Georgia Tech", "Wake Forest", "Auburn", "Alabama", "LSU",
    "Illinois", "Ohio State", "Colorado", "BYU",
]
CONFERENCES = {
    "UC Santa Barbara": "Big West", "Pepperdine": "WCC", "San Diego State": "MW",
    "USC": "Big Ten", "UCLA": "Big Ten", "Stanford": "ACC", "Cal": "ACC",
    "Arizona State": "Big 12", "Arizona": "Big 12", "Oregon": "Big Ten",
    "Washington": "Big Ten", "Texas": "SEC", "Oklahoma State": "Big 12",
    "Vanderbilt": "SEC", "Florida": "SEC", "Georgia Tech": "ACC",
    "Wake Forest": "ACC", "Auburn": "SEC", "Alabama": "SEC", "LSU": "SEC",
    "Illinois": "Big Ten", "Ohio State": "Big Ten", "Colorado": "Big 12",
    "BYU": "Big 12",
}
PLAYERS_PER_TEAM = 8
N_COLLEGE_TOURNAMENTS = 22

# ----------------------------------------------------------- junior config
N_JUNIORS = 170
N_JUNIOR_TOURNAMENTS = 36
JUNIOR_TOURS = {
    # tour: (tier_low, tier_high, typical rounds)
    "AJGA":            (3.6, 5.0, 3),
    "JGS Nationals":   (3.0, 4.2, 2),
    "FCG":             (2.0, 3.2, 2),
    "Toyota Tour Cup": (2.0, 3.2, 2),
    "SDJGA":           (1.0, 2.2, 2),
    "HSJGA":           (1.0, 2.0, 2),
    "NCJGA":           (1.2, 2.4, 2),
}
JUNIOR_CITIES = [
    ("San Diego", "CA", 32.72, -117.16), ("Carlsbad", "CA", 33.16, -117.35),
    ("Irvine", "CA", 33.68, -117.83), ("Scottsdale", "AZ", 33.49, -111.93),
    ("Las Vegas", "NV", 36.17, -115.14), ("Dallas", "TX", 32.78, -96.80),
    ("Orlando", "FL", 28.54, -81.38), ("Honolulu", "HI", 21.31, -157.86),
    ("Sacramento", "CA", 38.58, -121.49), ("Portland", "OR", 45.52, -122.68),
]

FIRST = ["Jake", "Ryan", "Tyler", "Luke", "Ben", "Sam", "Chris", "Nick", "Matt",
         "Josh", "Ethan", "Owen", "Cole", "Blake", "Drew", "Carson", "Mason",
         "Logan", "Hunter", "Trevor", "Aiden", "Connor", "Dylan", "Grant",
         "Kai", "Miles", "Jaden", "Preston", "Nolan", "Bryce"]
LAST = ["Anderson", "Miller", "Thompson", "Garcia", "Kim", "Nguyen", "Park",
        "Johnson", "Smith", "Brown", "Davis", "Wilson", "Moore", "Taylor",
        "Clark", "Lewis", "Walker", "Hall", "Young", "Allen", "Scott",
        "Baker", "Reed", "Cooper", "Tanaka", "Rivera", "Chang", "Okafor"]
COURSES = [
    ("Sandpiper GC", "Goleta", "CA", 34.42, -119.87, 72),
    ("Torrey Pines North", "La Jolla", "CA", 32.90, -117.25, 72),
    ("Karsten Creek", "Stillwater", "OK", 36.10, -97.14, 72),
    ("Riviera CC", "Pacific Palisades", "CA", 34.05, -118.50, 71),
    ("Bandon Trails", "Bandon", "OR", 43.11, -124.40, 71),
    ("Grayhawk GC", "Scottsdale", "AZ", 33.68, -111.90, 72),
    ("Colonial CC", "Fort Worth", "TX", 32.71, -97.36, 70),
    ("Shoal Creek", "Birmingham", "AL", 33.36, -86.66, 72),
    ("Olympia Fields", "Olympia Fields", "IL", 41.51, -87.70, 70),
    ("Chambers Bay", "University Place", "WA", 47.20, -122.57, 72),
    ("Pasatiempo GC", "Santa Cruz", "CA", 37.00, -122.02, 70),
    ("Admiral Baker GC", "San Diego", "CA", 32.79, -117.09, 72),
    ("Encinitas Ranch", "Encinitas", "CA", 33.06, -117.26, 72),
    ("Morgan Run", "Rancho Santa Fe", "CA", 33.01, -117.19, 71),
    ("Sterling Hills", "Camarillo", "CA", 34.23, -119.01, 71),
]


def _traits():
    """Sample per-player performance traits (mostly zero = no trait)."""
    big_game = rng.normal(0.9, 0.25) if rng.random() < 0.15 else (
        rng.normal(-0.9, 0.25) if rng.random() < 0.10 else 0.0)
    closer = rng.normal(1.2, 0.3) if rng.random() < 0.15 else (
        rng.normal(-1.0, 0.3) if rng.random() < 0.10 else 0.0)
    fast_start = rng.normal(1.0, 0.3) if rng.random() < 0.12 else 0.0
    wind_sens = float(np.clip(rng.normal(1.0, 0.35), 0.2, 2.0))
    return big_game, closer, fast_start, wind_sens


def _junior_trajectory():
    """(type, slope per career round, extras)"""
    u = rng.random()
    if u < 0.12:
        return "rapid_improver", float(rng.normal(-0.12, 0.02)), {}
    if u < 0.32:
        return "improver", float(rng.normal(-0.05, 0.015)), {}
    if u < 0.70:
        return "steady", float(rng.normal(0.0, 0.01)), {}
    if u < 0.82:
        return "plateau", float(rng.normal(-0.08, 0.02)), {"plateau_at": 28}
    if u < 0.92:
        return "decliner", float(rng.normal(0.06, 0.02)), {}
    # fluke season: a hot ~18-round stretch, then reversion
    start = int(rng.integers(8, 18))
    return "fluke", 0.0, {"fluke_start": start, "fluke_len": 18,
                          "fluke_boost": float(rng.normal(-3.2, 0.5))}


def make_players() -> pd.DataFrame:
    rows, pid = [], 1
    # ---- college
    for team in TEAMS:
        team_strength = rng.normal(0, 1.0)
        for _ in range(PLAYERS_PER_TEAM):
            skill = float(np.clip(rng.normal(2.5, 2.0) - team_strength, -2.5, 9.0))
            u = rng.random()
            sigma = (rng.uniform(1.6, 2.3) if u < 0.30 else
                     rng.uniform(3.6, 5.0) if u < 0.55 else
                     rng.uniform(2.4, 3.5))
            bg, cl, fs, ws = _traits()
            rows.append(dict(
                player_id=pid, player_name=f"{rng.choice(FIRST)} {rng.choice(LAST)}",
                level="college", team=team, conference=CONFERENCES[team],
                class_year=str(rng.choice(["FR", "SO", "JR", "SR"])),
                _skill=skill, _sigma=sigma,
                _traj="steady", _slope=float(rng.normal(-0.01, 0.02)), _tx={},
                _big_game=bg, _closer=cl, _fast=fs, _wind=ws,
                _injury_at=int(rng.integers(8, 16)) if rng.random() < 0.05 else -1,
            ))
            pid += 1
    # ---- juniors
    for _ in range(N_JUNIORS):
        city = JUNIOR_CITIES[int(rng.integers(0, len(JUNIOR_CITIES)))]
        skill = float(np.clip(rng.normal(6.5, 3.5), -1.5, 16.0))
        sigma = float(rng.uniform(2.2, 5.5))
        traj, slope, tx = _junior_trajectory()
        bg, cl, fs, ws = _traits()
        # circuit level: better juniors play stronger tours
        home_tier = float(np.clip(4.6 - skill * 0.35 + rng.normal(0, 0.5), 1.0, 5.0))
        rows.append(dict(
            player_id=pid, player_name=f"{rng.choice(FIRST)} {rng.choice(LAST)}",
            level="junior", team=f"{city[0]}, {city[1]}", conference="Junior",
            class_year=str(rng.choice(["2026", "2027", "2028", "2029"])),
            _skill=skill, _sigma=sigma, _traj=traj, _slope=slope, _tx=tx,
            _big_game=bg, _closer=cl, _fast=fs, _wind=ws,
            _injury_at=-1, _home_tier=home_tier,
        ))
        pid += 1
    return pd.DataFrame(rows)


def make_tournaments() -> pd.DataFrame:
    rows, tid = [], 1
    # ---- college slate
    date = SEASON_START
    for _ in range(N_COLLEGE_TOURNAMENTS):
        name, city, state, lat, lon, par = COURSES[int(rng.integers(0, 11))]
        difficulty = float(rng.normal(1.5, 1.3))
        rows.append(dict(
            tournament_id=tid,
            tournament_name=f"{name.split()[0]} Invitational" if tid % 3 else f"{city} Classic",
            level="college", tour="NCAA",
            event_tier=float(np.clip(rng.normal(4.2, 0.5), 3.2, 5.0)),
            start_date=date.date().isoformat(),
            course_name=name, city=city, state=state, latitude=lat, longitude=lon,
            par=par,
            course_rating=round(par + difficulty + rng.normal(0.5, 0.4), 1),
            slope=int(np.clip(rng.normal(134, 6), 120, 150)),
            num_rounds=3, _difficulty=difficulty,
        ))
        tid += 1
        date += pd.Timedelta(days=int(rng.integers(9, 15)))
    # ---- junior slate
    tours = list(JUNIOR_TOURS.keys())
    date = SEASON_START + pd.Timedelta(days=3)
    for _ in range(N_JUNIOR_TOURNAMENTS):
        tour = tours[int(rng.integers(0, len(tours)))]
        lo, hi, nr = JUNIOR_TOURS[tour]
        name, city, state, lat, lon, par = COURSES[int(rng.integers(0, len(COURSES)))]
        difficulty = float(rng.normal(0.2, 1.2))   # junior setups a bit easier
        rows.append(dict(
            tournament_id=tid,
            tournament_name=f"{tour} {city} {'Championship' if tid % 4 else 'Junior Open'}",
            level="junior", tour=tour,
            event_tier=float(rng.uniform(lo, hi)),
            start_date=date.date().isoformat(),
            course_name=name, city=city, state=state, latitude=lat, longitude=lon,
            par=par,
            course_rating=round(par + difficulty + rng.normal(-0.3, 0.4), 1),
            slope=int(np.clip(rng.normal(127, 7), 113, 148)),
            num_rounds=nr, _difficulty=difficulty,
        ))
        tid += 1
        date += pd.Timedelta(days=int(rng.integers(5, 9)))
    return pd.DataFrame(rows)


def _round_score(p, t, career_idx, round_num, wind, precip):
    """Latent score model — everything the ML layer tries to recover."""
    # trajectory
    if p._traj == "plateau":
        k = p._tx.get("plateau_at", 28)
        traj_adj = p._slope * min(career_idx, k)
    elif p._traj == "fluke":
        s, ln, boost = p._tx["fluke_start"], p._tx["fluke_len"], p._tx["fluke_boost"]
        traj_adj = boost if s <= career_idx < s + ln else 0.0
    else:
        traj_adj = p._slope * career_idx
    injury_adj = 3.5 if (p._injury_at >= 0 and career_idx >= p._injury_at * 3) else 0.0
    # traits
    tier_adj = -p._big_game * (t.event_tier - 3.0)
    closer_adj = -p._closer if round_num == t.num_rounds and t.num_rounds > 1 else 0.0
    fast_adj = -p._fast if round_num == 1 else 0.0
    wind_penalty = max(0.0, (wind - 10) * 0.12) * p._wind
    rain_penalty = precip * 2.5
    return (t.par + p._skill + t._difficulty + traj_adj + injury_adj
            + tier_adj + closer_adj + fast_adj + wind_penalty + rain_penalty
            + rng.normal(0, p._sigma))


def make_rounds(players: pd.DataFrame, tournaments: pd.DataFrame) -> pd.DataFrame:
    rows, rid = [], 1
    career = {int(p): 0 for p in players.player_id}
    college = players[players.level == "college"]
    juniors = players[players.level == "junior"]

    for _, t in tournaments.sort_values("start_date").iterrows():
        if t.level == "college":
            field_teams = rng.choice(TEAMS, size=int(len(TEAMS) * 0.55), replace=False)
            field = college[college.team.isin(field_teams)]
            field = (field.assign(_r=rng.random(len(field)))
                          .sort_values(["team", "_skill", "_r"])
                          .groupby("team").head(5))
        else:
            # juniors enter events near their circuit level
            gap = (juniors._home_tier - t.event_tier).abs()
            pool = juniors[gap < 1.3]
            n = min(len(pool), int(rng.integers(36, 52)))
            field = pool.sample(n=n, random_state=int(rng.integers(0, 1e6)))

        start = pd.Timestamp(t.start_date)
        # daily weather shared by the field
        for rnd in range(1, t.num_rounds + 1):
            rdate = start + pd.Timedelta(days=rnd - 1)
            temp = float(np.clip(rng.normal(66, 12), 34, 98))
            wind = float(np.clip(rng.gamma(2.2, 4.0), 1, 35))
            precip = float(max(0.0, rng.normal(-0.05, 0.12)))
            for _, p in field.iterrows():
                score = _round_score(p, t, career[int(p.player_id)], rnd, wind, precip)
                rows.append(dict(
                    round_id=rid, player_id=int(p.player_id),
                    tournament_id=int(t.tournament_id),
                    round_num=rnd, round_date=rdate.date().isoformat(),
                    score=int(round(np.clip(score, t.par - 10, t.par + 24))),
                    temp_f=round(temp, 1), wind_mph=round(wind, 1),
                    precip_in=round(precip, 2),
                ))
                rid += 1
        for pidx in field.player_id:
            career[int(pidx)] += int(t.num_rounds)
    return pd.DataFrame(rows)


def main():
    players = make_players()
    tournaments = make_tournaments()
    rounds = make_rounds(players, tournaments)

    pub = lambda df: df.drop(columns=[c for c in df.columns if c.startswith("_")])
    pub(players).to_csv(RAW_DIR / "players.csv", index=False)
    pub(tournaments).to_csv(RAW_DIR / "tournaments.csv", index=False)
    rounds.to_csv(RAW_DIR / "rounds.csv", index=False)

    jc = players.level.value_counts()
    print(f"players:     {len(players):>6,}  (college {jc.get('college',0)}, junior {jc.get('junior',0)})")
    print(f"tournaments: {len(tournaments):>6,}")
    print(f"rounds:      {len(rounds):>6,}")
    print(f"-> {RAW_DIR}")


if __name__ == "__main__":
    main()
