"""
BREAKOUT-FOOTBALL-PREDICTOR-2026-2027
--------------------------------------------------
SportPesa-inspired AI Value & Probability Predictor
- Dual-calendar date filtering
- Cumulative bankroll & Profit/Loss tracking (KES)
- High-contrast SportPesa design with team emblems
"""

import datetime
import json
import os
import pickle
import requests
import streamlit as st

# ---------------------------------------------------------
# Page Setup & High-Contrast SportPesa Branding Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="BREAKOUT-FOOTBALL-PREDICTOR-2026-2027",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Dark SportPesa-inspired Background */
    .stApp { background-color: #020B14; color: #FFFFFF; }
    
    /* High contrast text */
    h1, h2, h3, h4, h5, h6, p, label, span, div, small, strong {
        color: #FFFFFF !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0A192F !important;
        border-right: 2px solid #00D4FF;
    }
    
    /* Match Cards */
    .sp-card {
        background-color: #0F233D;
        border-radius: 10px;
        padding: 20px;
        border-left: 6px solid #00D4FF;
        border-top: 1px solid #1E3A5F;
        border-right: 1px solid #1E3A5F;
        border-bottom: 1px solid #1E3A5F;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.4);
    }
    
    /* Bankroll & Profit Tracking Box */
    .tracker-box {
        background-color: #0F233D;
        border: 2px solid #00D4FF;
        border-radius: 8px;
        padding: 15px;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    
    .tracker-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 6px;
        font-size: 14px;
    }
    
    /* Inputs */
    div[data-baseweb="input"] input {
        background-color: #020B14 !important;
        color: #FFFFFF !important;
        border: 1px solid #00D4FF !important;
        border-radius: 6px;
        font-weight: bold;
    }
    
    /* Custom Badges */
    .badge-prob {
        background-color: #00D4FF; color: #020B14 !important;
        font-weight: 800; padding: 4px 10px; border-radius: 4px; font-size: 12px;
        text-transform: uppercase;
    }
    .badge-val {
        background-color: #FFB703; color: #020B14 !important;
        font-weight: 800; padding: 4px 10px; border-radius: 4px; font-size: 12px;
        text-transform: uppercase;
    }
    .badge-market {
        background-color: #8D99AE; color: #020B14 !important;
        font-weight: 800; padding: 3px 8px; border-radius: 4px; font-size: 11px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRACKER_FILE = os.path.join(BASE_DIR, "clv_tracker_2026_27.json")

COVERED_LEAGUES = {
    "Premier League": 39,
    "La Liga": 140,
    "Serie A": 135,
    "Bundesliga": 78,
    "Ligue 1": 61,
}

def current_season():
    today = datetime.date.today()
    return today.year if today.month >= 7 else today.year - 1

SEASON = current_season()

# ---------------------------------------------------------
# Persistent Tracker Management (Profit / Loss over time)
# ---------------------------------------------------------
def load_tracker():
    if os.path.exists(TRACKER_FILE):
        try:
            with open(TRACKER_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_tracker(entries):
    with open(TRACKER_FILE, "w") as f:
        json.dump(entries, f, indent=4)

def get_tracker_metrics():
    entries = load_tracker()
    settled = [e for e in entries if e.get("result") in ["WON", "LOST"]]
    total_staked = sum(e.get("stake", 0.0) for e in settled)
    total_returned = sum(e.get("payout", 0.0) for e in settled)
    net_profit = total_returned - total_staked
    roi = (net_profit / total_staked * 100) if total_staked > 0 else 0.0
    return {
        "settled_count": len(settled),
        "pending_count": len([e for e in entries if e.get("result") == "PENDING"]),
        "total_staked": total_staked,
        "total_returned": total_returned,
        "net_profit": net_profit,
        "roi": roi,
    }

# ---------------------------------------------------------
# Model Artifacts & Odds Data Handling
# ---------------------------------------------------------
@st.cache_resource
def load_model_artifacts():
    artifacts = {"params": None, "home_adv": None, "calibration": None}
    p_path = os.path.join(BASE_DIR, "v9_params.json")
    if os.path.exists(p_path):
        with open(p_path) as f:
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
    path = os.path.join(BASE_DIR, "team_ratings.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def dixon_coles_probabilities(home_team, away_team, league, ratings, artifacts):
    league_ratings = ratings.get(league, {})
    if home_team not in league_ratings or away_team not in league_ratings:
        return None
    from scipy.stats import poisson
    home, away = league_ratings[home_team], league_ratings[away_team]
    ha = artifacts["home_adv"].get(league, {}).get("new_ha", 1.0) if artifacts["home_adv"] else 1.0
    rho = artifacts["params"]["dixon_coles"]["rho_domestic"] if artifacts["params"] else 0.0

    lam_home = home["attack"] * away["defense"] * ha
    lam_away = away["attack"] * home["defense"]

    max_goals = 10
    home_probs = [poisson.pmf(i, lam_home) for i in range(max_goals)]
    away_probs = [poisson.pmf(j, lam_away) for j in range(max_goals)]

    p_home = p_draw = p_away = 0.0
    for i in range(max_goals):
        for j in range(max_goals):
            p = home_probs[i] * away_probs[j]
            if i == 0 and j == 0: p *= (1 - lam_home * lam_away * rho)
            elif i == 0 and j == 1: p *= (1 + lam_home * rho)
            elif i == 1 and j == 0: p *= (1 + lam_away * rho)
            elif i == 1 and j == 1: p *= (1 - rho)

            if i > j: p_home += p
            elif i == j: p_draw += p
            else: p_away += p

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
    if not (odds_home and odds_draw and odds_away): return None
    inv = [1 / odds_home, 1 / odds_draw, 1 / odds_away]
    overround = sum(inv)
    return tuple(x / overround for x in inv)

@st.cache_data(ttl=900)
def load_live_verified_fixtures(start_d, end_d):
    fetch_ts = datetime.datetime.now(datetime.timezone.utc)
    try: api_key = st.secrets.get("API_SPORTS_KEY", "")
    except Exception: api_key = ""

    if not api_key or api_key == "your_actual_api_sports_key_here":
        return [], "MISSING_KEY", fetch_ts

    url = "https://v3.football.api-sports.io/fixtures"
    headers = {"x-apisports-key": api_key}
    verified_fixtures = []

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

        if "errors" in data and data["errors"]: continue

        for item in data.get("response", []):
            status_short = item["fixture"]["status"]["short"]
            if status_short not in ["NS", "TBD"]: continue

            fixture_dt = datetime.datetime.fromisoformat(
                item["fixture"]["date"].replace("Z", "+00:00")
            )
            verified_fixtures.append({
                "id": item["fixture"]["id"],
                "date": fixture_dt.date(),
                "time": fixture_dt.strftime("%H:%M"),
                "league": league_name,
                "country": item["league"]["country"],
                "home_team": item["teams"]["home"]["name"],
                "away_team": item["teams"]["away"]["name"],
                "home_logo": item["teams"]["home"]["logo"],
                "away_logo": item["teams"]["away"]["logo"],
            })
    return verified_fixtures, "OK", fetch_ts

@st.cache_data(ttl=600)
def load_fixture_odds(fixture_id):
    try: api_key = st.secrets.get("API_SPORTS_KEY", "")
    except Exception: api_key = ""
    if not api_key: return None

    url = "https://v3.football.api-sports.io/odds"
    headers = {"x-apisports-key": api_key}
    try:
        resp = requests.get(url, headers=headers, params={"fixture": fixture_id}, timeout=10)
        data = resp.json()
        books = data.get("response", [])
        if not books: return None

        home_odds, draw_odds, away_odds = [], [], []
        for entry in books:
            for bookmaker in entry.get("bookmakers", []):
                for bet in bookmaker.get("bets", []):
                    if bet["name"] == "Match Winner":
                        vals = {v["value"]: float(v["odd"]) for v in bet["values"]}
                        if "Home" in vals: home_odds.append(vals["Home"])
                        if "Draw" in vals: draw_odds.append(vals["Draw"])
                        if "Away" in vals: away_odds.append(vals["Away"])

        if not (home_odds and draw_odds and away_odds): return None
        avg = lambda lst: sum(lst) / len(lst)
        return avg(home_odds), avg(draw_odds), avg(away_odds)
    except Exception:
        return None

# ---------------------------------------------------------
# Main Interface Layout
# ---------------------------------------------------------
def main():
    st.sidebar.title("⚽ BREAKOUT 2026-2027")
    st.sidebar.markdown("---")

    # 1. Total Daily / Total Bankroll Impute (KES)
    st.sidebar.markdown("### 🇰🇪 Bankroll & Stake Allocation")
    total_bankroll_kes = st.sidebar.number_input(
        "Total Stake Budget (100% KES)",
        min_value=0.0,
        value=5000.0,
        step=100.0,
        format="%.2f",
        help="Input your starting/allocated 100% bankroll in Kenya Shillings."
    )

    metrics = get_tracker_metrics()
    current_net_pnl = metrics["net_profit"]
    pnl_color = "#2EC4B6" if current_net_pnl >= 0 else "#E63946"

    st.sidebar.markdown(
        f"""
        <div class="tracker-box">
            <h4 style="margin:0 0 8px 0; color:#00D4FF !important;">📊 Performance Tracker</h4>
            <div class="tracker-row"><span>Base Budget:</span><strong>KES {total_bankroll_kes:,.2f}</strong></div>
            <div class="tracker-row"><span>Total Settled Bets:</span><strong>{metrics['settled_count']}</strong></div>
            <div class="tracker-row"><span>Total Staked:</span><strong>KES {metrics['total_staked']:,.2f}</strong></div>
            <div class="tracker-row"><span>Total Payouts:</span><strong>KES {metrics['total_returned']:,.2f}</strong></div>
            <hr style="border: 0.5px solid #1E3A5F; margin: 8px 0;">
            <div class="tracker-row" style="font-size: 15px;">
                <span>Net Profit / Loss:</span>
                <strong style="color: {pnl_color} !important;">
                    KES {current_net_pnl:+,.2f} ({metrics['roi']:+.1f}% ROI)
                </strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Dual Calendar Date Selection
    st.sidebar.markdown("### 📅 Date Range Selectors")
    c_d1, c_d2 = st.sidebar.columns(2)
    today = datetime.date.today()
    start_date = c_d1.date_input("From Date", value=today, format="DD/MM/YYYY")
    end_date = c_d2.date_input("To Date", value=today + datetime.timedelta(days=1), format="DD/MM/YYYY")

    if start_date > end_date:
        st.sidebar.error("Error: 'From Date' must be earlier than or equal to 'To Date'.")

    # Fetch Fixtures
    fixtures, status_msg, fetched_at = load_live_verified_fixtures(start_date, end_date)
    artifacts = load_model_artifacts()
    team_ratings = load_team_ratings()

    # 3. League Filter Selection Options
    st.sidebar.markdown("### 🏆 Fixture Display Options")
    filter_mode = st.sidebar.radio(
        "Show Games:",
        ["All Available Games", "Filter per League"]
    )

    available_leagues = sorted(list(set(g["league"] for g in fixtures))) if fixtures else []
    
    if filter_mode == "Filter per League" and available_leagues:
        selected_league = st.sidebar.selectbox("Choose League", available_leagues)
        display_fixtures = [g for g in fixtures if g["league"] == selected_league]
    else:
        display_fixtures = fixtures

    # Main Header Dashboard
    st.title("⚽ BREAKOUT-FOOTBALL-PREDICTOR-2026-2027")
    st.markdown(f"**Selected Range:** `{start_date.strftime('%d/%m/%Y')}` to `{end_date.strftime('%d/%m/%Y')}`")
    
    # Top Metrics Cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("100% Stake Budget", f"KES {total_bankroll_kes:,.2f}")
    m2.metric("Cumulative P/L", f"KES {current_net_pnl:+,.2f}")
    m3.metric("Overall ROI", f"{metrics['roi']:+.1f}%")
    m4.metric("Games Displayed", len(display_fixtures))

    st.markdown("---")

    tab_games, tab_tracker = st.tabs(["🔥 Match Fixtures & Model Picks", "📝 Bet Outcome & Result Settle Log"])

    # ---------------------------------------------------------
    # TAB 1: Match Display with Emblems & Model Predictions
    # ---------------------------------------------------------
    with tab_games:
        if status_msg == "MISSING_KEY":
            st.warning("⚠️ Please configure `API_SPORTS_KEY` in Streamlit Secrets to load live games.")
        elif not display_fixtures:
            st.info("No fixtures available within the specified date range.")
        else:
            for g in display_fixtures:
                odds = load_fixture_odds(g["id"])
                model_probs = dixon_coles_probabilities(
                    g["home_team"], g["away_team"], g["league"], team_ratings, artifacts
                )

                if model_probs:
                    probs_tuple, source = model_probs, "model"
                elif odds:
                    probs_tuple, source = devig_probabilities(*odds), "market"
                else:
                    probs_tuple, source = None, None

                # Render SportPesa Match Card
                st.markdown('<div class="sp-card">', unsafe_allow_html=True)
                
                # Header info
                st.markdown(f"**📅 {g['date'].strftime('%d/%m/%Y')} @ {g['time']} EAT** | 🏆 **{g['league']}** ({g['country']})")
                
                c_teams, c_high_prob, c_proposed = st.columns([3, 3.5, 3.5])

                # Teams with Emblems
                with c_teams:
                    st.write("")
                    col_h, col_vs, col_a = st.columns([1, 0.5, 1])
                    if g["home_logo"]: col_h.image(g["home_logo"], width=48)
                    col_vs.markdown("<p style='text-align:center; padding-top:15px; font-weight:bold;'>VS</p>", unsafe_allow_html=True)
                    if g["away_logo"]: col_a.image(g["away_logo"], width=48)
                    st.markdown(f"<h4 style='text-align:center;'>{g['home_team']} <br><small>vs</small><br> {g['away_team']}</h4>", unsafe_allow_html=True)

                if not probs_tuple or not odds:
                    with c_high_prob:
                        st.warning("Odds/Model data pending for this fixture.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    continue

                probs = {
                    "Home Win": (probs_tuple[0], odds[0]),
                    "Draw": (probs_tuple[1], odds[1]),
                    "Away Win": (probs_tuple[2], odds[2]),
                }

                # Highest Probability Selection
                highest_prob_pick = max(probs, key=lambda k: probs[k][0])
                highest_prob_val = probs[highest_prob_pick][0] * 100
                highest_prob_odds = probs[highest_prob_pick][1]

                # Proposed Bet (Value Pick via EV)
                ev_scores = {k: (v[0] * v[1]) - 1 for k, v in probs.items()}
                proposed_pick = max(ev_scores, key=ev_scores.get)
                proposed_prob = probs[proposed_pick][0] * 100
                proposed_odds = probs[proposed_pick][1]

                badge_tag = f'<span class="badge-market">{source.upper()}</span>' if source else ""

                # Column 2: Highest Win Probability
                with c_high_prob:
                    st.markdown(f'<span class="badge-prob">Highest Win Probability</span> {badge_tag}', unsafe_allow_html=True)
                    st.markdown(f"### **{highest_prob_pick}**")
                    st.markdown(f"• Win Probability: **{highest_prob_val:.1f}%**")
                    st.markdown(f"• Decimal Odds: **{highest_prob_odds:.2f}**")

                # Column 3: Proposed Bet (Model Pick) & Actual Bet Impute Form
                with c_proposed:
                    st.markdown(f'<span class="badge-val">Proposed Bet (Model Pick)</span> {badge_tag}', unsafe_allow_html=True)
                    st.markdown(f"### **{proposed_pick}**")
                    st.markdown(f"• Model Odds: **{proposed_odds:.2f}** (Prob: **{proposed_prob:.1f}%**)")
                    
                    # Form to allow bettor to impute actual bet placed
                    with st.form(key=f"place_bet_form_{g['id']}"):
                        user_bet_selection = st.selectbox("Actual Pick Selected", ["Home Win", "Draw", "Away Win"], index=["Home Win", "Draw", "Away Win"].index(proposed_pick))
                        user_stake_kes = st.number_input("Actual Bet Placed (KES)", min_value=0.0, value=0.0, step=50.0)
                        submit_bet = st.form_submit_button("📌 Log Bet to Slip")
                        
                        if submit_bet and user_stake_kes > 0:
                            entries = load_tracker()
                            selected_odds = probs[user_bet_selection][1]
                            new_entry = {
                                "id": len(entries) + 1,
                                "date": g["date"].strftime("%d/%m/%Y"),
                                "league": g["league"],
                                "match": f"{g['home_team']} vs {g['away_team']}",
                                "highest_prob_bet": f"{highest_prob_pick} ({highest_prob_val:.1f}%)",
                                "proposed_bet": proposed_pick,
                                "bet_type": user_bet_selection,
                                "taken_odds": float(selected_odds),
                                "stake": float(user_stake_kes),
                                "payout": 0.0,
                                "profit": 0.0,
                                "result": "PENDING"
                            }
                            entries.append(new_entry)
                            save_tracker(entries)
                            st.success(f"Bet logged! KES {user_stake_kes:,.2f} on {user_bet_selection}")

                st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # TAB 2: Bet Log, Actual Outcomes, Amount Won/Lost
    # ---------------------------------------------------------
    with tab_tracker:
        st.subheader("📝 Bet History & Actual Outcome Settlement")
        entries = load_tracker()

        if not entries:
            st.info("No bets recorded yet. Place bets from the Fixtures tab.")
        else:
            pending_entries = [e for e in entries if e.get("result") == "PENDING"]
            
            # Settlement Form
            if pending_entries:
                st.markdown("### ⏳ Settle Pending Bets")
                with st.form("settle_bet_form"):
                    pending_labels = {f"#{e['id']} | {e['date']} - {e['match']} ({e['bet_type']}) - Stake: KES {e['stake']}": e['id'] for e in pending_entries}
                    selected_label = st.selectbox("Select Pending Bet to Settle", list(pending_labels.keys()))
                    chosen_id = pending_labels[selected_label]
                    
                    outcome = st.radio("Actual Outcome with Respect to Bet", ["WON", "LOST"])
                    
                    # Find chosen bet details for default payout
                    chosen_bet = next(e for e in entries if e["id"] == chosen_id)
                    default_payout = (chosen_bet["stake"] * chosen_bet["taken_odds"]) if outcome == "WON" else 0.0
                    
                    actual_payout = st.number_input("Actual Payout Amount (KES)", min_value=0.0, value=default_payout, step=50.0)
                    submit_settlement = st.form_submit_button("Check & Save Result")

                    if submit_settlement:
                        for e in entries:
                            if e["id"] == chosen_id:
                                e["result"] = outcome
                                e["payout"] = float(actual_payout)
                                e["profit"] = float(actual_payout) - float(e["stake"])
                                break
                        save_tracker(entries)
                        st.success(f"Settled Bet #{chosen_id} as {outcome}. Updated overall profit/loss!")
                        st.rerun()

            st.markdown("---")
            st.markdown("### 📋 Complete History & Profit/Loss Log")

            # Formatted Table
            table_data = []
            for e in entries:
                table_data.append({
                    "ID": e.get("id"),
                    "Date": e.get("date"),
                    "League": e.get("league"),
                    "Match": e.get("match"),
                    "Highest Prob Bet": e.get("highest_prob_bet", "N/A"),
                    "Proposed Pick": e.get("proposed_bet", "N/A"),
                    "Actual Pick Placed": e.get("bet_type"),
                    "Odds": f"{e.get('taken_odds'):.2f}",
                    "Stake (KES)": f"KES {e.get('stake'):,.2f}",
                    "Result": e.get("result"),
                    "Payout (KES)": f"KES {e.get('payout', 0.0):,.2f}",
                    "Net Profit/Loss (KES)": f"KES {e.get('profit', 0.0):+,.2f}",
                })
            
            st.dataframe(table_data, use_container_width=True)

if __name__ == "__main__":
    main()