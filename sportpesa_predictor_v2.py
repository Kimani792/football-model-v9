"""
SportPesa AI Value Predictor -- v2

Changes vs. the uploaded draft (see chat for full list):
  1. Removed hardcoded fake probabilities/odds that were identical for every fixture.
  2. Bet Tracker (Tab 3) now actually persists to clv_tracker_2026_27.json instead of
     discarding the result after each rerun.
  3. Fixed invalid CSS (`justify-between` -> `justify-content: space-between`).
  4. API call is now restricted to the 5 leagues your model covers + a season param,
     instead of fetching every fixture worldwide (quota burn).
  5. st.secrets access wrapped so a missing secrets.toml doesn't crash the app.
  6. Wired in the real V9 model artifacts (Dixon-Coles home advantage, rho,
     isotonic calibration, Kelly cap) instead of leaving them unused.

Still open / NOT faked:
  - True per-team probabilities require the rolling-xG team ratings file
    (team_ratings.json below) -- this hasn't been supplied yet. Until it is,
    the app falls back to de-vigged market-implied probability from real
    fetched odds, and says so on screen. It does NOT invent numbers.
"""

import datetime
import json
import os
import pickle

import requests
import streamlit as st

# ---------------------------------------------------------
# Page Setup & High-Contrast SportPesa Dark Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="SportPesa AI Predictor | Kenya",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #080e18; color: #ffffff; }
    h1, h2, h3, h4, h5, h6, p, label, span, div, small, strong {
        color: #ffffff !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    section[data-testid="stSidebar"] {
        background-color: #0f1926 !important;
        border-right: 2px solid #1e2d42;
    }
    .sp-card {
        background-color: #121e2e;
        border-radius: 8px;
        padding: 18px;
        border-left: 5px solid #00d4ff;
        border-top: 1px solid #1e2d42;
        border-right: 1px solid #1e2d42;
        border-bottom: 1px solid #1e2d42;
        margin-bottom: 16px;
    }
    .tracker-box {
        background-color: #16253b;
        border: 2px solid #00d4ff;
        border-radius: 8px;
        padding: 12px;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    .tracker-row {
        display: flex;
        justify-content: space-between;   /* FIX: was invalid "justify-between" */
        margin-bottom: 4px;
    }
    div[data-baseweb="input"] input {
        background-color: #080e18 !important;
        color: #ffffff !important;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-prob {
        background-color: #00d4ff; color: #000000 !important;
        font-weight: 800; padding: 4px 8px; border-radius: 4px; font-size: 12px;
    }
    .badge-val {
        background-color: #ffb703; color: #000000 !important;
        font-weight: 800; padding: 4px 8px; border-radius: 4px; font-size: 12px;
    }
    .badge-market {
        background-color: #8d99ae; color: #000000 !important;
        font-weight: 800; padding: 3px 7px; border-radius: 4px; font-size: 11px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# FIX: restrict to the leagues your model actually covers (matches
# league_home_advantage.json) instead of fetching every fixture worldwide.
# API-SPORTS league IDs, 2026 season.
COVERED_LEAGUES = {
    "Premier League": 39,
    "La Liga": 140,
    "Serie A": 135,
    "Bundesliga": 78,
    "Ligue 1": 61,
}
def current_season():
    """
    API-SPORTS labels a European season by its start year (e.g. the 2026/27
    season is queried as season=2026). Seasons start ~July/August, so any
    date from July onward belongs to the season starting that year; earlier
    months belong to the season that started the previous year.
    FIX: this was previously hardcoded to 2026 and would silently go stale
    at the next season rollover instead of tracking the real calendar.
    """
    today = datetime.date.today()
    return today.year if today.month >= 7 else today.year - 1


SEASON = current_season()


# ---------------------------------------------------------
# V9 Model Artifacts (Dixon-Coles params, calibration, home advantage)
# ---------------------------------------------------------
@st.cache_resource
def load_model_artifacts():
    artifacts = {"params": None, "home_adv": None, "calibration": None}

    params_path = os.path.join(BASE_DIR, "v9_params.json")
    if os.path.exists(params_path):
        with open(params_path) as f:
            artifacts["params"] = json.load(f)

    ha_path = os.path.join(BASE_DIR, "league_home_advantage.json")
    if os.path.exists(ha_path):
        with open(ha_path) as f:
            artifacts["home_adv"] = json.load(f)

    calib_path = os.path.join(BASE_DIR, "isotonic_calibration.pkl")
    if os.path.exists(calib_path):
        with open(calib_path, "rb") as f:
            artifacts["calibration"] = pickle.load(f)

    return artifacts


@st.cache_data(ttl=3600)
def load_team_ratings():
    """
    Per-team attack/defense strength (rolling-xG based), keyed by league then team.
    Schema:
        { "La Liga": { "Real Madrid": {"attack": 1.55, "defense": 0.80}, ... }, ... }
    NOT YET SUPPLIED. Returns {} until team_ratings.json exists alongside this
    script -- the app falls back to market-implied probability in that case
    rather than fabricating numbers.
    """
    path = os.path.join(BASE_DIR, "team_ratings.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def dixon_coles_probabilities(home_team, away_team, league, ratings, artifacts):
    """Returns (p_home, p_draw, p_away) from the real model, or None if the
    ratings needed for these two teams aren't available."""
    league_ratings = ratings.get(league, {})
    if home_team not in league_ratings or away_team not in league_ratings:
        return None

    from scipy.stats import poisson

    home = league_ratings[home_team]
    away = league_ratings[away_team]

    ha = artifacts["home_adv"].get(league, {}).get("new_ha", 1.0) if artifacts["home_adv"] else 1.0
    rho = artifacts["params"]["dixon_coles"]["rho_domestic"] if artifacts["params"] else 0.0

    lam_home = home["attack"] * away["defense"] * ha
    lam_away = away["attack"] * home["defense"]

    max_goals = 10
    home_probs = [poisson.pmf(i, lam_home) for i in range(max_goals)]
    away_probs = [poisson.pmf(i, lam_away) for i in range(max_goals)]

    p_home = p_draw = p_away = 0.0
    for i in range(max_goals):
        for j in range(max_goals):
            p = home_probs[i] * away_probs[j]
            # light Dixon-Coles low-score correlation adjustment
            if i == 0 and j == 0:
                p *= (1 - lam_home * lam_away * rho)
            elif i == 0 and j == 1:
                p *= (1 + lam_home * rho)
            elif i == 1 and j == 0:
                p *= (1 + lam_away * rho)
            elif i == 1 and j == 1:
                p *= (1 - rho)
            if i > j:
                p_home += p
            elif i == j:
                p_draw += p
            else:
                p_away += p

    total = p_home + p_draw + p_away
    p_home, p_draw, p_away = p_home / total, p_draw / total, p_away / total

    calib = artifacts.get("calibration")
    if calib:
        p_home = float(calib["home"].predict([p_home])[0])
        p_draw = float(calib["draw"].predict([p_draw])[0])
        p_away = float(calib["away"].predict([p_away])[0])
        s = p_home + p_draw + p_away
        p_home, p_draw, p_away = p_home / s, p_draw / s, p_away / s

    return p_home, p_draw, p_away


def devig_probabilities(odds_home, odds_draw, odds_away):
    """Market-implied probability with the overround removed. Used as an
    honest fallback when we don't have model ratings for a matchup."""
    if not (odds_home and odds_draw and odds_away):
        return None
    inv = [1 / odds_home, 1 / odds_draw, 1 / odds_away]
    overround = sum(inv)
    return tuple(x / overround for x in inv)


# ---------------------------------------------------------
# Live Fixtures Loader (API-SPORTS)
# ---------------------------------------------------------
@st.cache_data(ttl=900)
def load_live_verified_fixtures(start_d, end_d):
    fetch_ts = datetime.datetime.now(datetime.timezone.utc)
    try:
        api_key = st.secrets.get("API_SPORTS_KEY", "")
    except Exception:
        # FIX: st.secrets can raise if no secrets.toml exists at all
        api_key = ""

    if not api_key or api_key == "your_actual_api_sports_key_here":
        return [], "MISSING_KEY", fetch_ts

    url = "https://v3.football.api-sports.io/fixtures"
    headers = {"x-apisports-key": api_key}
    verified_fixtures = []

    # FIX: query per covered league (+ season) instead of one unfiltered
    # worldwide date-range call.
    for league_name, league_id in COVERED_LEAGUES.items():
        params = {
            "league": league_id,
            "season": SEASON,
            "from": start_d.strftime("%Y-%m-%d"),
            "to": end_d.strftime("%Y-%m-%d"),
            "timezone": "Africa/Nairobi",
        }
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            data = response.json()
        except Exception as e:
            return [], str(e), fetch_ts

        if "errors" in data and data["errors"]:
            continue  # skip this league on error, don't kill the whole load

        for item in data.get("response", []):
            status_short = item["fixture"]["status"]["short"]
            # FIX: this was a blocklist (only 3 excluded statuses), so
            # in-progress/finished matches from the same date window still
            # showed up as "predictions". Only show matches not yet kicked
            # off, since that's the only state a pre-match value bet makes
            # sense for.
            if status_short not in ["NS", "TBD"]:
                continue

            fixture_dt = datetime.datetime.fromisoformat(
                item["fixture"]["date"].replace("Z", "+00:00")
            )

            verified_fixtures.append(
                {
                    "id": item["fixture"]["id"],
                    "date": fixture_dt.date(),
                    "time": fixture_dt.strftime("%H:%M"),
                    "league": league_name,
                    "country": item["league"]["country"],
                    "home_team": item["teams"]["home"]["name"],
                    "away_team": item["teams"]["away"]["name"],
                    "home_logo": item["teams"]["home"]["logo"],
                    "away_logo": item["teams"]["away"]["logo"],
                    "status": item["fixture"]["status"]["long"],
                }
            )

    return verified_fixtures, "OK", fetch_ts


@st.cache_data(ttl=600)
def load_fixture_odds(fixture_id):
    """Real odds for a single fixture from API-SPORTS /odds. Returns None if
    unavailable -- callers must handle that, never substitute a fake number."""
    try:
        api_key = st.secrets.get("API_SPORTS_KEY", "")
    except Exception:
        api_key = ""
    if not api_key:
        return None

    url = "https://v3.football.api-sports.io/odds"
    headers = {"x-apisports-key": api_key}
    try:
        resp = requests.get(url, headers=headers, params={"fixture": fixture_id}, timeout=10)
        data = resp.json()
        books = data.get("response", [])
        if not books:
            return None
        # FIX: previously only checked the first bookmaker of the first
        # response entry. If that bookmaker didn't carry "Match Winner" (not
        # all do, for lower-profile fixtures) the odds silently came back
        # None even though other bookmakers had it. Now scans all of them
        # and averages across bookmakers that quote it, which is also a
        # steadier closing-line proxy for CLV purposes.
        home_odds, draw_odds, away_odds = [], [], []
        for entry in books:
            for bookmaker in entry.get("bookmakers", []):
                for bet in bookmaker.get("bets", []):
                    if bet["name"] == "Match Winner":
                        vals = {v["value"]: float(v["odd"]) for v in bet["values"]}
                        if "Home" in vals:
                            home_odds.append(vals["Home"])
                        if "Draw" in vals:
                            draw_odds.append(vals["Draw"])
                        if "Away" in vals:
                            away_odds.append(vals["Away"])
        if not (home_odds and draw_odds and away_odds):
            return None
        avg = lambda lst: sum(lst) / len(lst)
        return avg(home_odds), avg(draw_odds), avg(away_odds)
    except Exception:
        return None


# ---------------------------------------------------------
# CLV / Bet Tracker persistence (FIX: this used to just vanish on rerun)
# ---------------------------------------------------------
def load_tracker():
    path = os.path.join(BASE_DIR, "clv_tracker_2026_27.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def save_tracker(entries):
    path = os.path.join(BASE_DIR, "clv_tracker_2026_27.json")
    with open(path, "w") as f:
        json.dump(entries, f, indent=4)


def log_bet_with_clv(entries, league, match, bet_type, taken_odds, closing_odds, stake):
    """
    Merged in from the old football_predictor_v9.py AutomatedCLVTracker.
    Unlike that version, this appends to the SAME entries list/file the rest
    of this app uses, instead of a second independent tracker instance
    writing to the same path (the collision risk flagged earlier).
    closing_odds is optional at log time -- CLV can't be computed until a
    real closing line exists, so clv_pct is left null rather than computed
    against a placeholder.
    """
    clv_pct = round(((taken_odds / closing_odds) - 1.0) * 100.0, 2) if closing_odds else None
    entry = {
        "id": (max((e["id"] for e in entries), default=0) + 1),
        "date": datetime.date.today().isoformat(),
        "league": league.upper(),
        "match": match,
        "bet_type": bet_type,
        "taken_odds": float(taken_odds),
        "closing_odds": float(closing_odds) if closing_odds else None,
        "clv_pct": clv_pct,
        "stake": float(stake),
        "payout": None,
        "profit": None,
        "result": "PENDING",
    }
    entries.append(entry)
    save_tracker(entries)
    return entry


# ---------------------------------------------------------
# Page execution -- wrapped in main() so the functions above (the actual
# prediction/staking logic) can be imported and unit-tested without
# triggering a full Streamlit page render. `streamlit run` still executes
# this normally since __name__ == "__main__" in that context.
# ---------------------------------------------------------
def main():
    _run_sidebar_and_page()


def _run_sidebar_and_page():
    st.sidebar.title("🇰🇪 SportPesa Control Panel")
    st.sidebar.markdown("---")

    user_bankroll = st.sidebar.number_input(
        "Total Daily Stake Budget (100% Bankroll in KES)",
        min_value=0.0, value=0.0, step=100.0, format="%.2f",
    )

    placed_stakes = [
        st.session_state[k] for k in st.session_state
        if (k.startswith("stk_p_") or k.startswith("stk_v_"))
    ]
    total_placed_kes = sum(placed_stakes) if placed_stakes else 0.0
    remaining_kes = user_bankroll - total_placed_kes

    st.sidebar.markdown(
        f"""
        <div class="tracker-box">
            <h4 style="margin:0 0 8px 0; color:#00d4ff !important;">📊 Stake Tracker</h4>
            <div class="tracker-row"><span>100% Budget:</span><strong>KES {user_bankroll:,.2f}</strong></div>
            <div class="tracker-row" style="color:#ffb703 !important;">
                <span>Total Placed:</span><strong>KES {total_placed_kes:,.2f}</strong>
            </div>
            <hr style="border: 0.5px solid #1e2d42; margin: 6px 0;">
            <div class="tracker-row" style="font-size: 15px;">
                <span>Remaining:</span>
                <strong style="color: {'#2ec4b6' if remaining_kes >= 0 else '#e63946'} !important;">
                    KES {remaining_kes:,.2f}
                </strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("### 📅 Bet Date Filter")
    today_date = datetime.date.today()
    c_d1, c_d2 = st.sidebar.columns(2)
    start_date = c_d1.date_input("Start Date", value=today_date)
    end_date = c_d2.date_input("End Date", value=today_date + datetime.timedelta(days=1))
    if start_date > end_date:
        st.sidebar.error("Start Date cannot be later than End Date.")

    fixtures, status_msg, fetched_at = load_live_verified_fixtures(start_date, end_date)
    artifacts = load_model_artifacts()
    team_ratings = load_team_ratings()
    kelly_cap = 0.03  # from v9_params.json key_corrections_vs_v8
    if artifacts["params"]:
        kelly_cap = artifacts["params"].get("kelly_cap", 0.03)

    st.sidebar.markdown("### 🏆 League Filter")
    league_mode = st.sidebar.radio("Filter Games By:", ["All Leagues", "Specific League"])
    available_leagues = sorted(list(set(g["league"] for g in fixtures)))

    if league_mode == "Specific League" and available_leagues:
        selected_league = st.sidebar.selectbox("Choose League:", available_leagues)
        filtered_fixtures = [g for g in fixtures if g["league"] == selected_league]
    else:
        filtered_fixtures = fixtures

    if not team_ratings:
        st.sidebar.warning(
            "⚠️ No team_ratings.json found — predictions are falling back to "
            "de-vigged market-implied probability, not the Dixon-Coles model."
        )

    # ---------------------------------------------------------
    # Main Dashboard Header
    # ---------------------------------------------------------
    st.title("⚽ SportPesa AI Value Predictor")
    st.markdown(f"**Live Verified Fixtures:** {start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}")
    if status_msg == "OK":
        age_sec = (datetime.datetime.now(datetime.timezone.utc) - fetched_at).total_seconds()
        st.caption(
            f"🔄 Fixture/odds data fetched {int(age_sec)}s ago from API-SPORTS "
            f"(cache refreshes every 15 min for fixtures, 10 min for odds)."
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Target Bankroll (100%)", f"KES {user_bankroll:,.2f}")
    m2.metric("Stakes Placed So Far", f"KES {total_placed_kes:,.2f}")
    m3.metric("Remaining Unallocated", f"KES {remaining_kes:,.2f}")
    m4.metric("Verified Matches", len(filtered_fixtures))
    st.markdown("---")

    # ---------------------------------------------------------
    # Process Fixtures — real odds + real (or honestly-labeled fallback) probs
    # ---------------------------------------------------------
    processed_games = []
    for g in filtered_fixtures:
        odds = load_fixture_odds(g["id"])  # (home, draw, away) or None
        model_probs = dixon_coles_probabilities(
            g["home_team"], g["away_team"], g["league"], team_ratings, artifacts
        )

        source = None
        probs_tuple = None
        if model_probs:
            probs_tuple = model_probs
            source = "model"
        elif odds:
            probs_tuple = devig_probabilities(*odds)
            source = "market"

        if probs_tuple is None or odds is None:
            # Can't responsibly show a "value" bet without either a real model
            # probability or real odds. Skip the EV machinery for this fixture.
            processed_games.append({"game": g, "unavailable": True})
            continue

        probs = {
            "Home Win": (probs_tuple[0], odds[0]),
            "Draw": (probs_tuple[1], odds[1]),
            "Away Win": (probs_tuple[2], odds[2]),
        }

        best_prob_bet = max(probs, key=lambda k: probs[k][0])
        ev_scores = {k: (v[0] * v[1]) - 1 for k, v in probs.items()}
        best_val_bet = max(ev_scores, key=ev_scores.get)

        def kelly_stake(bet_key):
            p = probs[bet_key][0]
            b = probs[bet_key][1] - 1
            if b <= 0:
                return 0.0
            raw_kelly = max(0.0, (b * p - (1 - p)) / b)
            capped = min(raw_kelly, kelly_cap)  # FIX: apply v9 kelly_cap, not an arbitrary 0.25
            return user_bankroll * capped

        processed_games.append({
            "game": g,
            "unavailable": False,
            "source": source,
            "best_prob_bet": best_prob_bet,
            "best_prob_pct": probs[best_prob_bet][0] * 100,
            "best_prob_odds": probs[best_prob_bet][1],
            "rec_stake_prob": kelly_stake(best_prob_bet),
            "best_val_bet": best_val_bet,
            "best_val_pct": probs[best_val_bet][0] * 100,
            "best_val_odds": probs[best_val_bet][1],
            "rec_stake_val": kelly_stake(best_val_bet),
        })

    # ---------------------------------------------------------
    # Tabs
    # ---------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["🎯 Match Predictions", "🚀 Day's Accumulator", "📝 SportPesa Bet Slip"])

    with tab1:
        if status_msg == "MISSING_KEY":
            st.warning("⚠️ Please configure `API_SPORTS_KEY` in Streamlit Secrets to load live verified matches.")
        elif not filtered_fixtures:
            st.info("No verified games scheduled for the selected date range.")
        else:
            for item in processed_games:
                g = item["game"]
                st.markdown('<div class="sp-card">', unsafe_allow_html=True)
                st.markdown(f"**📅 {g['date'].strftime('%a, %b %d')} @ {g['time']} EAT** | 🏆 {g['league']} ({g['country']})")

                col_teams, col_prob, col_val = st.columns([2.5, 3.5, 3.5])
                with col_teams:
                    st.markdown("<br>", unsafe_allow_html=True)
                    lh, lvs, la = st.columns([1, 1, 1])
                    if g["home_logo"]:
                        lh.image(g["home_logo"], width=42)
                    lvs.markdown("**VS**")
                    if g["away_logo"]:
                        la.image(g["away_logo"], width=42)
                    st.markdown(f"**{g['home_team']}** vs **{g['away_team']}**")

                if item["unavailable"]:
                    st.info("No real odds and no model rating available for this fixture yet — skipped rather than guessed.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    continue

                badge = '<span class="badge-market">MARKET-IMPLIED</span>' if item["source"] == "market" else ""

                with col_prob:
                    st.markdown(f'<span class="badge-prob">HIGHEST PROBABILITY WIN</span> {badge}', unsafe_allow_html=True)
                    st.markdown(f"**Pick:** `{item['best_prob_bet']}`")
                    st.markdown(
                        f"• Win Prob: **{item['best_prob_pct']:.1f}%**\n\n"
                        f"• Odds: **{item['best_prob_odds']:.2f}**\n\n"
                        f"• Rec. Stake (Kelly, capped {kelly_cap*100:.0f}%): **KES {item['rec_stake_prob']:,.2f}**"
                    )
                    st.number_input("Stake Placed (KES)", min_value=0.0, value=0.0, step=50.0, key=f"stk_p_{g['id']}")

                with col_val:
                    st.markdown(f'<span class="badge-val">HIGHEST VALUE (+EV) BET</span> {badge}', unsafe_allow_html=True)
                    st.markdown(f"**Pick:** `{item['best_val_bet']}`")
                    st.markdown(
                        f"• Win Prob: **{item['best_val_pct']:.1f}%**\n\n"
                        f"• Odds: **{item['best_val_odds']:.2f}**\n\n"
                        f"• Rec. Stake (Kelly, capped {kelly_cap*100:.0f}%): **KES {item['rec_stake_val']:,.2f}**"
                    )
                    st.number_input("Stake Placed (KES)", min_value=0.0, value=0.0, step=50.0, key=f"stk_v_{g['id']}")

                st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.subheader("🔥 Day's Most Probable Accumulator Multi-Bet")
        usable_games = [p for p in processed_games if not p.get("unavailable")]
        if len(usable_games) < 2:
            st.warning("At least 2 fixtures with real odds/model data are required to form an accumulator.")
        else:
            acc_picks = sorted(usable_games, key=lambda x: x["best_prob_pct"], reverse=True)[:3]
            acc_total_odds = 1.0
            for p in acc_picks:
                acc_total_odds *= p["best_prob_odds"]
            acc_rec_stake = user_bankroll * 0.10

            st.markdown('<div class="sp-card">', unsafe_allow_html=True)
            st.markdown("### Recommended 3-Fold Safety Multi-Bet")
            for p in acc_picks:
                st.markdown(
                    f"• **{p['game']['home_team']} vs {p['game']['away_team']}** ➔ "
                    f"Bet: **{p['best_prob_bet']}** @ Odds `{p['best_prob_odds']:.2f}` "
                    f"(Win Prob: {p['best_prob_pct']:.1f}%)"
                )
            st.markdown("---")
            a1, a2, a3 = st.columns(3)
            a1.metric("Combined Multi Odds", f"{acc_total_odds:.2f}")
            a2.metric("Suggested Multi Stake", f"KES {acc_rec_stake:,.2f}" if user_bankroll > 0 else "Set Budget")
            a3.metric("Est. Payout", f"KES {acc_rec_stake * acc_total_odds:,.2f}" if user_bankroll > 0 else "KES 0.00")
            st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.subheader("📝 Bet & CLV Tracker")

        st.markdown("#### Log a new bet (at time of placing)")
        with st.form("clv_log_form"):
            c1, c2, c3 = st.columns(3)
            game_list = (
                [f"{g['home_team']} vs {g['away_team']} ({g['date']})" for g in filtered_fixtures]
                if filtered_fixtures else ["No Games Loaded"]
            )
            log_league = c1.selectbox("League", list(COVERED_LEAGUES.keys()) + ["OTHER"])
            log_match = c2.selectbox("Match", game_list)
            log_bet_type = c3.text_input("Selection", value="Home Win")

            d1, d2, d3 = st.columns(3)
            log_taken_odds = d1.number_input("Odds Taken", min_value=1.01, value=1.90, step=0.01)
            log_closing_odds = d2.number_input(
                "Closing Odds (0 = not known yet)", min_value=0.0, value=0.0, step=0.01
            )
            log_stake = d3.number_input("Stake (KES)", min_value=0.0, value=200.0, step=50.0)

            submit_log = st.form_submit_button("Log Bet")

        if submit_log:
            entries = load_tracker()
            entry = log_bet_with_clv(
                entries, log_league, log_match, log_bet_type,
                log_taken_odds, log_closing_odds if log_closing_odds > 0 else None, log_stake,
            )
            if entry["clv_pct"] is not None:
                st.success(f"Bet logged — CLV: {entry['clv_pct']:+.2f}%")
            else:
                st.success("Bet logged — CLV pending closing odds.")

        st.markdown("---")
        st.markdown("#### Settle a pending bet")
        entries = load_tracker()
        pending = [e for e in entries if e.get("result") == "PENDING"]

        if not pending:
            st.info("No pending bets to settle.")
        else:
            with st.form("settle_form"):
                pending_labels = [f"#{e['id']} — {e['match']} ({e['bet_type']})" for e in pending]
                pick = st.selectbox("Pending bet", pending_labels)
                picked_id = int(pick.split("—")[0].strip().lstrip("#"))
                settle_closing_odds = st.number_input(
                    "Closing odds (leave 0 if already set)", min_value=0.0, value=0.0, step=0.01
                )
                payout_amt = st.number_input("Payout received (KES, 0 if lost)", min_value=0.0, step=50.0)
                submit_settle = st.form_submit_button("Settle")

            if submit_settle:
                for e in entries:
                    if e["id"] == picked_id:
                        if settle_closing_odds > 0 and e.get("closing_odds") is None:
                            e["closing_odds"] = settle_closing_odds
                            e["clv_pct"] = round(((e["taken_odds"] / settle_closing_odds) - 1.0) * 100.0, 2)
                        e["payout"] = payout_amt
                        e["profit"] = payout_amt - e["stake"]
                        e["result"] = "WON" if payout_amt > 0 else "LOST"
                        break
                save_tracker(entries)
                st.success(f"Settled bet #{picked_id}.")

        st.markdown("---")
        st.markdown("#### Full Log")
        st.json(load_tracker())


if __name__ == "__main__":
    main()
