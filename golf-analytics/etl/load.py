"""
Load raw CSVs into staging tables, then build SQL marts.

Usage:  python etl/load.py
Reads:  data/raw/{players,tournaments,rounds}.csv
Writes: stg_* tables + mart_* tables in DATABASE_URL (sqlite by default)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sqlalchemy import create_engine, text

from config import DATABASE_URL, RAW_DIR, ROOT


def run_sql_file(conn, path: Path):
    raw = path.read_text()
    # strip line comments, split on ';'
    stmts = []
    for chunk in raw.split(";"):
        lines = [l for l in chunk.splitlines() if not l.strip().startswith("--")]
        stmt = "\n".join(lines).strip()
        if stmt:
            stmts.append(stmt)
    for stmt in stmts:
        conn.execute(text(stmt))


def main():
    engine = create_engine(DATABASE_URL)

    players = pd.read_csv(RAW_DIR / "players.csv")
    tournaments = pd.read_csv(RAW_DIR / "tournaments.csv")
    rounds = pd.read_csv(RAW_DIR / "rounds.csv")

    with engine.begin() as conn:
        print("Creating staging schema...")
        run_sql_file(conn, ROOT / "db" / "schema.sql")

    players.to_sql("stg_players", engine, if_exists="append", index=False)
    tournaments.to_sql("stg_tournaments", engine, if_exists="append", index=False)
    rounds.to_sql("stg_rounds", engine, if_exists="append", index=False)
    print(f"Loaded staging: {len(players)} players, {len(tournaments)} tournaments, "
          f"{len(rounds)} rounds")

    with engine.begin() as conn:
        print("Building marts (sql/marts.sql)...")
        run_sql_file(conn, ROOT / "sql" / "marts.sql")

    with engine.connect() as conn:
        for tbl in ["mart_player_rounds", "mart_field_strength",
                    "mart_player_summary", "mart_team_leaderboard"]:
            n = conn.execute(text(f"SELECT COUNT(*) FROM {tbl}")).scalar()
            print(f"  {tbl}: {n:,} rows")

    print("Done.")


if __name__ == "__main__":
    main()
