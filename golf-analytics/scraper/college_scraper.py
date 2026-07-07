"""
College golf data ingestion — targets Clippd Scoreboard (scoreboard.clippd.com).

WHY CLIPPD, NOT GOLFSTAT: Golfstat retired after losing the NCAA contract;
NCAA scoring/rankings moved (via Spikemark) to Clippd Scoreboard, which is
now the official source for college results. Good news: it is a modern
single-page app, which means the data arrives as JSON from API endpoints —
no HTML parsing needed once you find the endpoints.

HOW TO FIND THE ENDPOINTS (10 minutes, one time):
1. Open https://scoreboard.clippd.com in Chrome.
2. DevTools -> Network tab -> filter "Fetch/XHR".
3. Navigate to a tournament leaderboard, a team page, a player page.
4. Each click fires JSON requests — copy those URLs and response shapes
   into the TODO constants below. Typical patterns look like
   /api/tournaments/{id}/scoring, /api/players/{id}/results, etc.
5. Run on ONE tournament first, verify the CSVs, then scale up.

Also grab course rating/slope wherever the event page exposes them — the
junior<->college differential bridge depends on those two fields. If an
event lacks them, geocode the course and look rating/slope up separately
(cache aggressively).

Respect the site: low concurrency, delays, honest User-Agent, and read
Clippd's Terms of Service before scraping at scale — this is the NCAA's
official vendor, so consider emailing them for research access; a student
project pitch sometimes just gets you the data.

Output: same three CSVs as data/generate_sample_data.py — players.csv,
tournaments.csv, rounds.csv — so the rest of the pipeline is untouched.

Usage:  python scraper/college_scraper.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import pandas as pd

from config import RAW_DIR

BASE = "https://scoreboard.clippd.com"
CONCURRENCY = 3
DELAY_SECONDS = 1.0
HEADERS = {
    "User-Agent": "college-golf-analytics-research (student project; contact: your_email@ucsb.edu)",
    "Accept": "application/json",
}

# TODO: fill these in from DevTools -> Network -> Fetch/XHR
TOURNAMENT_LIST_ENDPOINT = f"{BASE}/api/..."     # list of events for a season
TOURNAMENT_ENDPOINT_TMPL = f"{BASE}/api/..."     # per-event scoring payload
PLAYER_ENDPOINT_TMPL = f"{BASE}/api/..."         # per-player results payload

semaphore = asyncio.Semaphore(CONCURRENCY)


async def fetch_json(client: httpx.AsyncClient, url: str) -> dict:
    async with semaphore:
        r = await client.get(url, headers=HEADERS, timeout=30,
                             follow_redirects=True)
        r.raise_for_status()
        await asyncio.sleep(DELAY_SECONDS)
        return r.json()


def parse_tournament(payload: dict) -> tuple[dict, list[dict]]:
    """TODO: map Clippd JSON -> (tournaments.csv record, rounds.csv records).

    tournaments.csv needs: tournament_id, tournament_name, level='college',
    tour='NCAA', event_tier, start_date, course_name, city, state,
    latitude, longitude, par, course_rating, slope, num_rounds.
    event_tier: derive from field strength or event class until a better
    signal exists (rankings-points weight if exposed).

    rounds.csv needs: round_id, player_id, tournament_id, round_num,
    round_date, score (weather columns left empty — backfilled by
    features/weather.py from Open-Meteo).
    """
    return {}, []


async def main():
    async with httpx.AsyncClient() as client:
        if "..." in TOURNAMENT_LIST_ENDPOINT:
            print(__doc__)
            print(">>> Endpoints not filled in yet — follow the DevTools "
                  "workflow in the docstring first.")
            return
        # payload = await fetch_json(client, TOURNAMENT_LIST_ENDPOINT)
        # ... iterate events -> parse -> write the three CSVs to RAW_DIR


if __name__ == "__main__":
    asyncio.run(main())
