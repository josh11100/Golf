"""
College + Junior Golf Performance Intelligence — Streamlit dashboard.

Usage:  streamlit run app/dashboard.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine

from config import ARTIFACTS_DIR, DATABASE_URL

st.set_page_config(page_title="Golf Performance Intelligence",
                   page_icon="⛳", layout="wide")

# Dark-first palette. Rules: one saturated accent (fairway green) carries
# meaning, gold is data, red is warning only. Text/lines never fall below
# ~4.5:1 contrast on the #0E1512 background.
ACCENT = "#52B788"    # bright fairway green — headings, form lines, active UI
GREEN = "#2D6A4F"     # deep green — fills, badges (white text on top)
SAND = "#E3B23C"      # warm gold — data points
CLAY = "#E07A5F"      # warning coral — declines, flags
MUTE = "#9BA8A0"      # muted sage — secondary text, gridlines
INK = "#E8E4D8"       # cream — primary text on dark
PALETTE = [ACCENT, SAND, CLAY, "#74C69D", "#C08552", "#83A598", "#D4A373"]
TRAJ_COLORS = {"Rapidly Improving": ACCENT, "Improving": "#74C69D",
               "Steady": MUTE, "Plateauing": SAND, "Declining": CLAY,
               "Insufficient data": "#666666"}

st.markdown(f"""
<style>
  h1, h2, h3 {{ color: {ACCENT}; letter-spacing: -0.01em; }}
  [data-testid="stMetricValue"] {{ color: {ACCENT}; }}
  [data-testid="stCaptionContainer"] {{ color: {MUTE}; }}
  /* Tabs: inactive = quiet card with readable muted text;
     active = solid accent with DARK text (highest contrast pairing) */
  .stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid #2A362F; }}
  .stTabs [data-baseweb="tab"] {{
      background: #1A241F; color: {MUTE}; border-radius: 8px 8px 0 0;
      padding: 10px 20px; border: 1px solid #2A362F; border-bottom: none;
      font-weight: 600; }}
  .stTabs [data-baseweb="tab"]:hover {{ color: {INK}; background: #22302A; }}
  .stTabs [aria-selected="true"] {{
      background: {ACCENT} !important; color: #0E1512 !important;
      border-color: {ACCENT}; }}
  .stTabs [aria-selected="true"] p {{ color: #0E1512 !important; }}
  .badge {{ display:inline-block; background:{GREEN}; color:{INK};
            border-radius:12px; padding:2px 12px; margin:2px 4px 2px 0;
            font-size:0.85em; }}
  .badge-neg {{ background:#7A3B2E; }}
  .demo-banner {{ background:#2A2410; border:1px solid {SAND}; color:{SAND};
            border-radius:8px; padding:8px 14px; font-size:0.9em;
            margin-bottom:10px; }}
</style>
""", unsafe_allow_html=True)

# All plotly charts inherit a dark template so axis/legend text is readable
import plotly.io as pio
pio.templates["golf_dark"] = go.layout.Template(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK),
        xaxis=dict(gridcolor="#243029", zerolinecolor="#243029"),
        yaxis=dict(gridcolor="#243029", zerolinecolor="#243029"),
        legend=dict(font=dict(color=INK)),
    ))
px.defaults.template = "golf_dark"


@st.cache_resource
def get_engine():
    return create_engine(DATABASE_URL)


@st.cache_data(ttl=600)
def load(table: str) -> pd.DataFrame:
    return pd.read_sql(f"SELECT * FROM {table}", get_engine())


def try_load(table: str):
    try:
        return load(table)
    except Exception:
        return None


rounds = try_load("mart_player_rounds")
summary = try_load("mart_player_summary")
teams = try_load("mart_team_leaderboard")
fields = try_load("mart_field_strength")
arch = try_load("player_archetypes")
anoms = try_load("anomaly_flags")
preds = try_load("predictions")
traits = try_load("player_traits")
traj = try_load("player_trajectory")
readiness = try_load("player_readiness")

if rounds is None or summary is None:
    st.error("Database is empty — run the pipeline (see README quick start).")
    st.stop()


def form_chart(player_id: int, y_col: str = "adj_score",
               roll_col: str = "rolling_adj_5", title: str = "",
               y_title: str = "Strokes vs field"):
    pr = (rounds[rounds.player_id == player_id]
          .sort_values(["round_date", "round_num"]).reset_index(drop=True))
    pr["idx"] = pr.index + 1
    fig = go.Figure(layout=dict(template="golf_dark"))
    fig.add_hline(y=0, line_dash="dot", line_color=MUTE)
    fig.add_trace(go.Scatter(
        x=pr.idx, y=pr[y_col], mode="markers", name="Round",
        marker=dict(color=SAND, size=7, opacity=0.7),
        hovertemplate="Round %{x}<br>%{y:+.1f}<br>%{customdata}",
        customdata=pr.tournament_name))
    if roll_col in pr:
        fig.add_trace(go.Scatter(x=pr.idx, y=pr[roll_col], mode="lines",
                                 name="Form", line=dict(color=ACCENT, width=3)))
    fig.update_layout(title=title, xaxis_title="Career round",
                      yaxis_title=y_title, height=400,
                      plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=1.1))
    return fig


def strengths_card(player_id: int):
    """Earned trait badges with the numbers behind them; silence if none."""
    if traits is None:
        return
    t = traits[(traits.player_id == player_id) & traits.badge.astype(bool)]
    if len(t) == 0:
        st.caption("No statistically earned trait badges yet — "
                   "not enough rounds in contrasting contexts.")
        return
    for _, r in t.iterrows():
        cls = "badge" if r.shrunk_effect > 0 else "badge badge-neg"
        st.markdown(
            f'<span class="{cls}">{r.description}</span> '
            f'<span style="color:#666;font-size:0.85em">'
            f'{r.shrunk_effect:+.1f} strokes · {r.n_high + r.n_low} rounds '
            f'(z={r.z:.1f})</span>', unsafe_allow_html=True)


def trajectory_line(player_id: int):
    if traj is None:
        return None
    t = traj[traj.player_id == player_id]
    return t.iloc[0] if len(t) else None


# ---------------------------------------------------------------- header
st.title("⛳ Golf Performance Intelligence")
st.caption("College + junior golf on one absolute scale — USGA differentials, "
           "field-adjusted scoring, trajectory detection, and recruiting readiness")
st.markdown('<div class="demo-banner">⚠️ <b>Demo dataset</b> — all players, '
            'teams, and results are synthetic placeholders used to develop the '
            'pipeline. Real rosters load once the Clippd Scoreboard / junior '
            'tour ingestion is connected.</div>', unsafe_allow_html=True)

n_col = summary[summary.level == "college"].player_id.nunique()
n_jun = summary[summary.level == "junior"].player_id.nunique()
c1, c2, c3, c4 = st.columns(4)
c1.metric("College players", f"{n_col:,}")
c2.metric("Junior players", f"{n_jun:,}")
c3.metric("Rounds analyzed", f"{len(rounds):,}")
c4.metric("Watchlist (rising)", f"{int(traj.watchlist.sum()) if traj is not None else '—'}")

tab_overview, tab_player, tab_recruit, tab_arch, tab_anom, tab_model = st.tabs(
    ["Season Overview", "Player Explorer", "Recruiting Board",
     "Archetypes", "Anomaly Watch", "Model Performance"])

# ---------------------------------------------------------------- overview
with tab_overview:
    level = st.radio("Level", ["college", "junior"], horizontal=True,
                     format_func=str.title)
    lvl = summary[summary.level == level]

    left, right = st.columns([3, 2])
    with left:
        st.subheader(f"{level.title()} leaderboard")
        st.caption("Differential = (score − course rating) × 113 / slope — "
                   "the absolute scale. Adj = strokes vs that day's field.")
        lb = (lvl.sort_values("avg_differential").head(25)
              [["player_name", "team", "class_year", "rounds_played",
                "avg_differential", "avg_adj_score", "adj_score_std"]]
              .rename(columns={"player_name": "Player", "team": "Team/Home",
                               "class_year": "Class", "rounds_played": "Rds",
                               "avg_differential": "Differential",
                               "avg_adj_score": "Adj Avg",
                               "adj_score_std": "Volatility"}))
        st.dataframe(lb.style.format({"Differential": "{:+.2f}",
                                      "Adj Avg": "{:+.2f}",
                                      "Volatility": "{:.2f}"}),
                     width="stretch", height=520, hide_index=True)

    with right:
        if level == "college" and teams is not None:
            st.subheader("Team standings")
            t = teams.sort_values("team_rank").head(15)
            fig = px.bar(t, x="team_avg_adj", y="team", orientation="h",
                         color_discrete_sequence=[ACCENT],
                         labels={"team_avg_adj": "Team avg adj. score", "team": ""})
            fig.update_layout(yaxis=dict(autorange="reversed"), height=430,
                              margin=dict(l=0, r=10, t=10, b=0),
                              plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")
        elif fields is not None:
            st.subheader("Junior events by tour")
            jt = fields[fields.level == "junior"]
            fig = px.scatter(jt, x="event_tier", y="field_strength",
                             color="tour", size="field_players",
                             hover_name="tournament_name",
                             color_discrete_sequence=PALETTE,
                             labels={"event_tier": "Event tier (1–5)",
                                     "field_strength": "Field strength (avg diff.)"})
            fig.update_layout(height=430, plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")

        st.subheader("Toughest fields")
        if fields is not None:
            f = (fields[fields.level == level]
                 .sort_values("field_strength").head(8))
            st.dataframe(
                f[["tournament_name", "tour", "field_players", "field_strength"]]
                .rename(columns={"tournament_name": "Tournament", "tour": "Tour",
                                 "field_players": "Players",
                                 "field_strength": "Field strength"})
                .style.format({"Field strength": "{:+.2f}"}),
                width="stretch", hide_index=True)

# ---------------------------------------------------------------- player
with tab_player:
    pcol1, pcol2 = st.columns([1, 3])
    with pcol1:
        lvl_sel = st.selectbox("Level", ["college", "junior"],
                               format_func=str.title, key="pe_level")
        pool = summary[summary.level == lvl_sel]
        team_sel = st.selectbox("Team / Home", sorted(pool.team.unique()))
        roster = pool[pool.team == team_sel].sort_values("avg_differential")
        player_sel = st.selectbox("Player", roster.player_name.tolist())
        prow = roster[roster.player_name == player_sel].iloc[0]

        st.metric("Differential avg", f"{prow.avg_differential:+.2f}")
        st.metric("Adj. average", f"{prow.avg_adj_score:+.2f}")
        st.metric("Volatility (σ)", f"{prow.adj_score_std:.2f}")

        tl = trajectory_line(prow.player_id)
        if tl is not None:
            st.markdown(f"**Trajectory:** "
                        f"<span style='color:{TRAJ_COLORS.get(tl.trajectory, INK)};"
                        f"font-weight:600'>{tl.trajectory}</span>",
                        unsafe_allow_html=True)
            if tl.recent_slope_per10 is not None and not pd.isna(tl.recent_slope_per10):
                st.caption(f"{tl.recent_slope_per10:+.2f} strokes / 10 rounds "
                           f"(z={tl.slope_z:.1f})")
            if bool(tl.fluke_flag):
                st.warning(f"Hot-streak reversion detected "
                           f"({tl.fluke_gap:+.1f} strokes better than "
                           f"surrounding play) — season averages may flatter.")
        if arch is not None:
            a = arch[arch.player_id == prow.player_id]
            if len(a):
                st.markdown(f"**Archetype:** {a.iloc[0].archetype}")

        st.markdown("**Strengths**")
        strengths_card(prow.player_id)

    with pcol2:
        st.plotly_chart(form_chart(prow.player_id,
                                   title=f"{player_sel} — season form "
                                         f"(field-adjusted)"),
                        width="stretch")
        pr = rounds[rounds.player_id == prow.player_id]
        wx = pr.dropna(subset=["wind_mph"])
        cA, cB = st.columns(2)
        with cA:
            if len(wx) > 5:
                fig2 = px.scatter(wx, x="wind_mph", y="adj_score",
                                  color_discrete_sequence=[ACCENT],
                                  labels={"wind_mph": "Wind (mph)",
                                          "adj_score": "Adj. score"},
                                  title="Wind sensitivity")
                fig2.update_layout(height=300, plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig2, width="stretch")
        with cB:
            if pr.event_tier.nunique() > 2:
                fig3 = px.scatter(pr, x="event_tier", y="differential",
                                  color_discrete_sequence=[SAND],
                                  labels={"event_tier": "Event tier",
                                          "differential": "Differential"},
                                  title="Performance vs competition level")
                fig3.update_layout(height=300, plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig3, width="stretch")

# ---------------------------------------------------------------- recruiting
with tab_recruit:
    st.subheader("Recruiting board")
    st.caption("Juniors and college players share one absolute scale (USGA "
               "differentials), so a junior's recent form places directly "
               "on the current D1 distribution.")

    rc1, rc2 = st.columns([2, 3])

    with rc1:
        st.markdown("#### 🔭 Coach's watchlist — rising fast")
        if traj is not None:
            wl = (traj[traj.watchlist.astype(bool) & (traj.level == "junior")]
                  .sort_values("recent_slope_per10"))
            if readiness is not None:
                wl = wl.merge(readiness[["player_id", "college_percentile",
                                         "readiness_band"]],
                              on="player_id", how="left")
            st.dataframe(
                wl[["player_name", "team", "class_year", "recent_slope_per10",
                    "recent_diff", "college_percentile"]]
                .rename(columns={"player_name": "Player", "team": "Home",
                                 "class_year": "Grad",
                                 "recent_slope_per10": "Trend (/10 rds)",
                                 "recent_diff": "Recent diff.",
                                 "college_percentile": "D1 %ile"})
                .style.format({"Trend (/10 rds)": "{:+.2f}",
                               "Recent diff.": "{:+.2f}",
                               "D1 %ile": "{:.0f}"}),
                width="stretch", hide_index=True, height=340)
            st.caption("Improvement measured on differentials — real gains, "
                       "not softer fields. Statistical trend only "
                       "(z ≤ −2.2), so a hot week can't buy a spot here.")

        st.markdown("#### ⚠️ Hot-streak reversions")
        if traj is not None:
            fl = traj[traj.fluke_flag.astype(bool)]
            st.dataframe(
                fl[["player_name", "level", "team", "fluke_gap", "trajectory"]]
                .rename(columns={"player_name": "Player", "level": "Level",
                                 "team": "Team/Home",
                                 "fluke_gap": "Streak vs baseline",
                                 "trajectory": "Current"})
                .style.format({"Streak vs baseline": "{:+.1f}"}),
                width="stretch", hide_index=True)
            st.caption("A stretch this much better than everything around it "
                       "reverted — rankings built on it deserve a second look.")

    with rc2:
        st.markdown("#### 🎓 College readiness")
        if readiness is None:
            st.info("Run `python models/readiness.py` first.")
        else:
            jsel = st.selectbox(
                "Junior",
                readiness.sort_values("college_percentile", ascending=False)
                         .player_name.tolist())
            jr = readiness[readiness.player_name == jsel].iloc[0]

            m1, m2, m3 = st.columns(3)
            m1.metric("D1 percentile", f"{jr.college_percentile:.0f}%")
            m2.metric("Recent differential", f"{jr.recent_diff_20:+.2f}")
            m3.metric("Band", jr.readiness_band.split(" range")[0],
                      help=jr.readiness_band)
            st.caption(f"Closest current college comp: "
                       f"**{jr.comparable_college_player}**")

            col_dist = summary[(summary.level == "college")
                               & summary.recent_diff_20.notna()]
            fig = go.Figure(layout=dict(template="golf_dark"))
            fig.add_trace(go.Histogram(
                x=col_dist.recent_diff_20, nbinsx=30, name="D1 players",
                marker_color=GREEN, opacity=0.75))
            fig.add_vline(x=jr.recent_diff_20, line_color=CLAY, line_width=3,
                          annotation_text=jsel, annotation_font_color=CLAY)
            fig.update_layout(
                title="Where this junior lands on the current D1 distribution",
                xaxis_title="Recent-20 differential (lower = better)",
                yaxis_title="College players", height=330,
                plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig, width="stretch")

            st.plotly_chart(
                form_chart(jr.player_id, y_col="differential",
                           roll_col="rolling_diff_10",
                           title=f"{jsel} — differential trajectory",
                           y_title="Differential"),
                width="stretch")
            st.markdown("**Strengths**")
            strengths_card(jr.player_id)

# ---------------------------------------------------------------- archetypes
with tab_arch:
    st.subheader("Player archetypes — KMeans on season profile")
    if arch is None:
        st.info("Run `python models/player_archetypes.py` first.")
    else:
        arch2 = arch.merge(summary[["player_id", "level"]], on="player_id",
                           how="left")
        lv = st.radio("Level", ["all", "college", "junior"], horizontal=True,
                      format_func=str.title, key="arch_level")
        if lv != "all":
            arch2 = arch2[arch2.level == lv]
        fig = px.scatter(
            arch2, x="avg_adj_score", y="adj_score_std", color="archetype",
            size="rounds_played", hover_name="player_name",
            hover_data={"team": True, "season_trend": ":.2f"},
            color_discrete_sequence=PALETTE,
            labels={"avg_adj_score": "Adjusted scoring average (lower = better)",
                    "adj_score_std": "Volatility (σ)"})
        fig.update_layout(height=520, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- anomalies
with tab_anom:
    st.subheader("Anomaly watch — sustained performance drops")
    st.caption("Changepoint scan + IsolationForest consensus. A sustained "
               "collapse, not one bad week — the 'something's off' signal.")
    if anoms is None:
        st.info("Run `python models/anomaly_detection.py` first.")
    else:
        flagged = anoms[anoms.flagged.astype(bool)].sort_values(
            "z_score", ascending=False)
        st.metric("Players flagged", len(flagged))
        st.dataframe(
            flagged[["player_name", "team", "changepoint_round",
                     "baseline_adj", "recent_adj", "drop_strokes", "z_score"]]
            .rename(columns={"player_name": "Player", "team": "Team/Home",
                             "changepoint_round": "Break at round",
                             "baseline_adj": "Before", "recent_adj": "After",
                             "drop_strokes": "Drop", "z_score": "z"})
            .style.format({"Before": "{:+.2f}", "After": "{:+.2f}",
                           "Drop": "{:+.2f}", "z": "{:.2f}"}),
            width="stretch", hide_index=True)
        if len(flagged):
            sel = st.selectbox("Inspect", flagged.player_name)
            pid = flagged[flagged.player_name == sel].iloc[0].player_id
            st.plotly_chart(form_chart(pid, title=f"{sel} — the break"),
                            width="stretch")

# ---------------------------------------------------------------- model
with tab_model:
    st.subheader("XGBoost round prediction — holdout performance")
    metrics_path = ARTIFACTS_DIR / "xgb_metrics.json"
    if preds is None or not metrics_path.exists():
        st.info("Run `python models/train_performance.py` first.")
    else:
        m = json.loads(metrics_path.read_text())
        a, b, c = st.columns(3)
        a.metric("MAE (strokes)", f"{m['mae']:.2f}",
                 delta=f"{m['mae'] - m['baseline_mae']:.2f} vs naive baseline",
                 delta_color="inverse")
        b.metric("R²", f"{m['r2']:.3f}")
        c.metric("Test rounds", f"{m['n_test']:,}",
                 help=f"Time-based split, cutoff {m['cutoff']}")
        left, right = st.columns(2)
        with left:
            fig = px.scatter(preds, x="predicted_adj", y="adj_score",
                             opacity=0.35, color_discrete_sequence=[ACCENT],
                             labels={"predicted_adj": "Predicted",
                                     "adj_score": "Actual"},
                             title="Predicted vs actual (test set)")
            lim = [preds.predicted_adj.min() - 1, preds.predicted_adj.max() + 1]
            fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                                     line=dict(color=MUTE, dash="dot"),
                                     showlegend=False))
            fig.update_layout(height=420, plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")
        with right:
            imp = pd.Series(m["feature_importance"]).sort_values().reset_index()
            imp.columns = ["feature", "importance"]
            fig = px.bar(imp, x="importance", y="feature", orientation="h",
                         color_discrete_sequence=[ACCENT],
                         title="Feature importance")
            fig.update_layout(height=420, plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")

st.divider()
st.caption("Scale: USGA differentials bridge junior ↔ college · Traits: "
           "shrinkage-tested badges · Trajectories: OLS trend + fluke scan · "
           "Weather: Open-Meteo / OpenWeather")
