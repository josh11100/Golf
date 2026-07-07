"""Central config. Reads .env — never hardcode secrets in code."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# sqlite for instant local dev; swap to postgres for the real build:
# DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/golf
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{ROOT / 'golf.db'}")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

RAW_DIR = ROOT / "data" / "raw"
ARTIFACTS_DIR = ROOT / "models" / "artifacts"
RAW_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
