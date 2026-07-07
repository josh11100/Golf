-- =====================================================================
-- College + Junior Golf Performance Intelligence — database schema
-- Staging layer: raw scraped/loaded data. Marts built by sql/marts.sql.
-- Portable across PostgreSQL and SQLite.
-- =====================================================================

DROP TABLE IF EXISTS stg_rounds;
DROP TABLE IF EXISTS stg_tournaments;
DROP TABLE IF EXISTS stg_players;

CREATE TABLE stg_players (
    player_id     INTEGER PRIMARY KEY,
    player_name   TEXT NOT NULL,
    level         TEXT NOT NULL DEFAULT 'college',  -- 'college' | 'junior'
    team          TEXT NOT NULL,       -- college: school | junior: hometown
    conference    TEXT,                -- college: conf   | junior: 'Junior'
    class_year    TEXT                 -- college: FR/SO/JR/SR | junior: grad year
);

CREATE TABLE stg_tournaments (
    tournament_id   INTEGER PRIMARY KEY,
    tournament_name TEXT NOT NULL,
    level           TEXT NOT NULL DEFAULT 'college',
    tour            TEXT,               -- NCAA | AJGA | JGS | FCG | Toyota Tour Cup
                                        -- | SDJGA | HSJGA | NCJGA | ...
    event_tier      REAL,               -- unified 1-5 competition level
                                        -- (from AJGA points tiers / JGS difficulty
                                        --  ranks / college field strength)
    start_date      DATE NOT NULL,
    course_name     TEXT,
    city            TEXT,
    state           TEXT,
    latitude        REAL,
    longitude       REAL,
    par             INTEGER NOT NULL DEFAULT 72,
    course_rating   REAL,               -- USGA rating — the junior<->college bridge
    slope           INTEGER,            -- USGA slope
    num_rounds      INTEGER NOT NULL DEFAULT 3
);

CREATE TABLE stg_rounds (
    round_id      INTEGER PRIMARY KEY,
    player_id     INTEGER NOT NULL REFERENCES stg_players(player_id),
    tournament_id INTEGER NOT NULL REFERENCES stg_tournaments(tournament_id),
    round_num     INTEGER NOT NULL,
    round_date    DATE NOT NULL,
    score         INTEGER NOT NULL,
    temp_f        REAL,
    wind_mph      REAL,
    precip_in     REAL
);

CREATE INDEX idx_rounds_player  ON stg_rounds(player_id);
CREATE INDEX idx_rounds_tourney ON stg_rounds(tournament_id);
CREATE INDEX idx_rounds_date    ON stg_rounds(round_date);
