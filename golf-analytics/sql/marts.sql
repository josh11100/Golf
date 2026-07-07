-- =====================================================================
-- Marts layer — the SQL-depth showcase.
-- CTEs + window functions, portable across PostgreSQL and SQLite.
-- Two comparison scales:
--   adj_score    = score - field average (within-population, per day)
--   differential = (score - course_rating) * 113 / slope  (USGA math —
--                  the ABSOLUTE scale that bridges juniors and college)
-- =====================================================================

-- ---------------------------------------------------------------------
-- mart_player_rounds: one row per player-round, fully enriched
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS mart_player_rounds;

CREATE TABLE mart_player_rounds AS
WITH round_field AS (
    SELECT
        r.round_id, r.player_id, r.tournament_id, r.round_num, r.round_date,
        r.score, r.temp_f, r.wind_mph, r.precip_in,
        t.par, t.course_name, t.tournament_name,
        t.level, t.tour, t.event_tier, t.course_rating, t.slope, t.num_rounds,
        r.score - t.par                                            AS score_to_par,
        (r.score - t.course_rating) * 113.0 / t.slope              AS differential,
        AVG(r.score) OVER (PARTITION BY r.tournament_id, r.round_num)  AS field_avg,
        COUNT(*)     OVER (PARTITION BY r.tournament_id, r.round_num)  AS field_size,
        RANK() OVER (PARTITION BY r.tournament_id, r.round_num
                     ORDER BY r.score)                                 AS field_rank
    FROM stg_rounds r
    JOIN stg_tournaments t ON t.tournament_id = r.tournament_id
),
adjusted AS (
    SELECT *, score - field_avg AS adj_score
    FROM round_field
),
form AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY player_id
                           ORDER BY round_date, round_num)             AS career_round_num,
        AVG(adj_score) OVER (PARTITION BY player_id
                             ORDER BY round_date, round_num
                             ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS rolling_adj_5,
        AVG(differential) OVER (PARTITION BY player_id
                                ORDER BY round_date, round_num
                                ROWS BETWEEN 9 PRECEDING AND 0 PRECEDING) AS rolling_diff_10,
        LAG(adj_score) OVER (PARTITION BY player_id
                             ORDER BY round_date, round_num)           AS prev_adj
    FROM adjusted
)
SELECT *, adj_score - prev_adj AS adj_delta
FROM form;

-- ---------------------------------------------------------------------
-- mart_field_strength: tournament field quality (within-level)
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS mart_field_strength;

CREATE TABLE mart_field_strength AS
WITH player_season AS (
    SELECT player_id, AVG(differential) AS season_diff
    FROM mart_player_rounds
    GROUP BY player_id
),
entrants AS (
    SELECT DISTINCT r.tournament_id, r.player_id FROM stg_rounds r
)
SELECT
    e.tournament_id, t.tournament_name, t.level, t.tour, t.event_tier,
    t.start_date,
    COUNT(*)            AS field_players,
    AVG(ps.season_diff) AS field_strength,
    RANK() OVER (PARTITION BY t.level
                 ORDER BY AVG(ps.season_diff)) AS field_strength_rank
FROM entrants e
JOIN player_season ps  ON ps.player_id = e.player_id
JOIN stg_tournaments t ON t.tournament_id = e.tournament_id
GROUP BY e.tournament_id, t.tournament_name, t.level, t.tour,
         t.event_tier, t.start_date;

-- ---------------------------------------------------------------------
-- mart_player_summary: season profile per player
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS mart_player_summary;

CREATE TABLE mart_player_summary AS
WITH per_round AS (
    SELECT
        m.player_id, m.adj_score, m.differential, m.score_to_par,
        m.field_rank, m.field_size, m.career_round_num,
        CASE WHEN m.field_rank <= m.field_size * 0.10 THEN 1 ELSE 0 END AS top10pct
    FROM mart_player_rounds m
),
halves AS (
    SELECT
        player_id,
        AVG(CASE WHEN career_round_num <= 15 THEN adj_score END) AS early_adj,
        AVG(CASE WHEN career_round_num >  15 THEN adj_score END) AS late_adj
    FROM per_round
    GROUP BY player_id
),
recent AS (
    -- last-20-rounds differential: the recruiting readiness number
    SELECT player_id, AVG(differential) AS recent_diff_20
    FROM (
        SELECT player_id, differential,
               ROW_NUMBER() OVER (PARTITION BY player_id
                                  ORDER BY career_round_num DESC) AS rn
        FROM per_round
    ) x
    WHERE rn <= 20
    GROUP BY player_id
)
SELECT
    p.player_id, p.player_name, p.level, p.team, p.conference, p.class_year,
    COUNT(*)                 AS rounds_played,
    AVG(pr.score_to_par)     AS avg_to_par,
    AVG(pr.adj_score)        AS avg_adj_score,
    AVG(pr.differential)     AS avg_differential,
    rc.recent_diff_20,
    CASE WHEN COUNT(*) > 1 THEN
        SQRT(AVG(pr.adj_score * pr.adj_score) - AVG(pr.adj_score) * AVG(pr.adj_score))
    END                      AS adj_score_std,
    AVG(pr.top10pct * 1.0)   AS top10pct_rate,
    MIN(pr.differential)     AS best_differential,
    h.late_adj - h.early_adj AS season_trend
FROM per_round pr
JOIN stg_players p ON p.player_id = pr.player_id
JOIN halves h      ON h.player_id = pr.player_id
JOIN recent rc     ON rc.player_id = pr.player_id
GROUP BY p.player_id, p.player_name, p.level, p.team, p.conference,
         p.class_year, h.late_adj, h.early_adj, rc.recent_diff_20;

-- ---------------------------------------------------------------------
-- mart_team_leaderboard (college only)
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS mart_team_leaderboard;

CREATE TABLE mart_team_leaderboard AS
SELECT
    p.team, p.conference,
    COUNT(DISTINCT m.player_id)             AS players,
    COUNT(*)                                AS rounds,
    AVG(m.adj_score)                        AS team_avg_adj,
    AVG(m.differential)                     AS team_avg_diff,
    RANK() OVER (ORDER BY AVG(m.adj_score)) AS team_rank
FROM mart_player_rounds m
JOIN stg_players p ON p.player_id = m.player_id
WHERE p.level = 'college'
GROUP BY p.team, p.conference;
