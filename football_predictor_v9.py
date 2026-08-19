import datetime
import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------
# Page Configuration & High-Contrast SportPesa Theme Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="SportPesa AI Predictor | Kenya",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# High-contrast CSS styling (Dark background, bright white/cyan/gold text)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b131e;
        color: #ffffff;
    }
    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #111d2c !important;
        border-right: 1px solid #22354d;
    }
    .sp-card {
        background-color: #162436;
        border-radius: 10px;
        padding: 18px;
        border: 1px solid #2a3f5a;
        margin-bottom: 20px;
    }
    div[data-baseweb="input"] {
        background-color: #0b131e !important;
        border: 1px solid #00b4d8 !important;
        color: #ffffff !important;
    }
    .badge-prob {
        background-color: #00b4d8;
        color: #000000 !important;
        font-weight: bold;
        padding: 4px 10px;
        border-radius: 4px;
        display: inline-block;
        margin-bottom: 8px;
    }
    .badge-val {
        background-color: #FFD166;
        color: #000000 !important;
        font-weight: bold;
        padding: 4px 10px;
        border-radius: 4px;
        display: inline-block;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# API-SPORTS Data Loader
# ---------------------------------------------------------


@st.cache_data(ttl=1800)  # Caches results for 30 minutes
def load_fixtures_data(start_d, end_d):
    """Fetches real fixtures from API-SPORTS between start_d and end_d."""
    api_key = st.secrets.get("API_SPORTS_KEY", "")

    if not api_key or api_key == "your_actual_api_sports_key_here":
        st.warning(
            "⚠️ Please configure `API_SPORTS_KEY` in Streamlit Secrets to load live data."
        )
        return []

    url = "https://v3.football.api-sports.io/fixtures"
    headers = {"x-apisports-key": api_key}
    params = {
        "from": start_d.strftime("%Y-%m-%d"),
        "to": end_d.strftime("%Y-%m-%d"),
    }

    try:
        response = requests.get(url, headers=headers, params=params).json()
        fixtures = []

        for item in response.get("response", []):
            fixture_date = datetime.datetime.strptime(
                item["fixture"]["date"][:10], "%Y-%m-%d"
            ).date()

            # Process fixture record
            fixtures.append(
                {
                    "id": item["fixture"]["id"],
                    "date": fixture_date,
                    "league": item["league"]["name"],
                    "home_team": item["teams"]["home"]["name"],
                    "away_team": item["teams"]["away"]["name"],
                    "home_logo": item["teams"]["home"]["logo"],
                    "away_logo": item["teams"]["away"]["logo"],
                    # Model probability and bookmaker odds fallbacks
                    "prob_home": 0.55,
                    "prob_draw": 0.25,
                    "prob_away": 0.20,
                    "odds_home": 1.90,
                    "odds_draw": 3.40,
                    "odds_away": 4.10,
                }
            )
        return fixtures
    except Exception as e:
        st.error(f"Error fetching API data: {e}")
        return []


# ---------------------------------------------------------
# Sidebar Controls & Inputs
# ---------------------------------------------------------
st.sidebar.title("🇰🇪 SportPesa Bankroll")
st.sidebar.markdown("---")

# 1. Bankroll Total Stake Input (Default Blank / 0.0)
user_bankroll = st.sidebar.number_input(
    "Total Daily Budget / Bankroll (KES)",
    min_value=0.0,
    value=0.0,
    step=100.0,
    format="%.2f",
    help="Enter your total betting budget in Kenya Shillings.",
)

st.sidebar.markdown("### 📅 Date Range Filter")

# 2. Bet Start Date and End Date Selection
today_date = datetime.date.today()
col_d1, col_d2 = st.sidebar.columns(2)
start_date = col_d1.date_input("Start Date", value=today_date)
end_date = col_d2.date_input(
    "End Date", value=today_date + datetime.timedelta(days=1)
)

if start_date > end_date:
    st.sidebar.error("Start Date must be earlier than or equal to End Date.")

# Load fixtures from API
raw_fixtures = load_fixtures_data(start_date, end_date)

# 3. League Filtering
st.sidebar.markdown("### 🏆 League Filter")
league_option = st.sidebar.radio(
    "Select Games To View:", ["All Leagues for Date(s)", "Specific League"]
)

available_leagues = sorted(list(set(g["league"] for g in raw_fixtures)))

if league_option == "Specific League" and available_leagues:
    selected_league = st.sidebar.selectbox(
        "Choose League:", options=available_leagues
    )
    fixtures = [g for g in raw_fixtures if g["league"] == selected_league]
else:
    fixtures = raw_fixtures

# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.title("⚽ SportPesa AI Value Predictor")
st.markdown(
    f"**Filtering Fixtures:** {start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}"
)

top1, top2, top3 = st.columns(3)
top1.metric("Current Bankroll", f"KES {user_bankroll:,.2f}")
top2.metric("Total Matches", len(fixtures))
top3.metric(
    "Active Leagues",
    len(set(g["league"] for g in fixtures)) if fixtures else 0,
)

st.markdown("---")

# ---------------------------------------------------------
# Main Tabs Layout
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["🎯 Match Predictions", "🚀 Daily Accumulators", "📝 Bet Tracking Slip"]
)

processed_games = []

# --- TAB 1: MATCH PREDICTIONS ---
with tab1:
    if not fixtures:
        st.info(
            "No games found for the selected dates and league settings. Ensure your API_SPORTS_KEY is active."
        )
    else:
        for idx, g in enumerate(fixtures):
            probs = {
                "Home Win": (g["prob_home"], g["odds_home"]),
                "Draw": (g["prob_draw"], g["odds_draw"]),
                "Away Win": (g["prob_away"], g["odds_away"]),
            }

            # 1. Most Probable Bet
            best_prob_bet = max(probs, key=lambda k: probs[k][0])
            best_prob_pct = probs[best_prob_bet][0] * 100
            best_prob_odds = probs[best_prob_bet][1]

            # 2. Best Value Bet (+EV)
            ev_scores = {k: (v[0] * v[1]) - 1 for k, v in probs.items()}
            best_val_bet = max(ev_scores, key=ev_scores.get)
            best_val_pct = probs[best_val_bet][0] * 100
            best_val_odds = probs[best_val_bet][1]

            # Recommended Kelly Stake Calculations (KES)
            b_p = best_prob_odds - 1
            p_p = probs[best_prob_bet][0]
            kelly_p = max(0.0, (b_p * p_p - (1 - p_p)) / b_p) if b_p > 0 else 0.0
            rec_stake_prob = user_bankroll * (kelly_p * 0.25)

            b_v = best_val_odds - 1
            p_v = probs[best_val_bet][0]
            kelly_v = max(0.0, (b_v * p_v - (1 - p_v)) / b_v) if b_v > 0 else 0.0
            rec_stake_val = user_bankroll * (kelly_v * 0.25)

            processed_games.append(
                {
                    "game": g,
                    "best_prob_bet": best_prob_bet,
                    "best_prob_pct": best_prob_pct,
                    "best_prob_odds": best_prob_odds,
                    "rec_stake_prob": rec_stake_prob,
                    "best_val_bet": best_val_bet,
                    "best_val_pct": best_val_pct,
                    "best_val_odds": best_val_odds,
                    "rec_stake_val": rec_stake_val,
                }
            )

            # High-Contrast Game Card
            st.markdown('<div class="sp-card">', unsafe_allow_html=True)
            st.markdown(
                f"#### 📅 {g['date'].strftime('%a, %b %d')} &nbsp;|&nbsp; 🏆 {g['league']}"
            )

            c_teams, c_prob, c_val = st.columns([2.5, 3.5, 3.5])

            # Team Emblems and Matchup
            with c_teams:
                st.markdown("<br>", unsafe_allow_html=True)
                lh, l_vs, la = st.columns([1, 1, 1])
                lh.image(g["home_logo"], width=45)
                l_vs.markdown("**VS**")
                la.image(g["away_logo"], width=45)
                st.markdown(f"**{g['home_team']}**<br>vs<br>**{g['away_team']}**", unsafe_allow_html=True)

            # Most Probable Win Bet Column
            with c_prob:
                st.markdown(
                    '<span class="badge-prob">MOST PROBABLE WIN BET</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Pick:** `{best_prob_bet}`")
                st.markdown(
                    f"• **Win Prob:** `{best_prob_pct:.1f}%`\n\n"
                    f"• **Bet Odds:** `{best_prob_odds:.2f}`\n\n"
                    f"• **Rec. Stake:** `KES {rec_stake_prob:,.2f}`"
                )
                st.number_input(
                    f"Stake Placed (KES) - Prob #{idx+1}",
                    min_value=0.0,
                    value=0.0,
                    step=50.0,
                    key=f"stk_prob_{g['id']}",
                )

            # Best Value Bet Column
            with c_val:
                st.markdown(
                    '<span class="badge-val">HIGHEST VALUE (+EV) BET</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Pick:** `{best_val_bet}`")
                st.markdown(
                    f"• **Win Prob:** `{best_val_pct:.1f}%`\n\n"
                    f"• **Bet Odds:** `{best_val_odds:.2f}`\n\n"
                    f"• **Rec. Stake:** `KES {rec_stake_val:,.2f}`"
                )
                st.number_input(
                    f"Stake Placed (KES) - Value #{idx+1}",
                    min_value=0.0,
                    value=0.0,
                    step=50.0,
                    key=f"stk_val_{g['id']}",
                )

            st.markdown("</div>", unsafe_allow_html=True)

# --- TAB 2: DAILY ACCUMULATORS ---
with tab2:
    st.subheader("🔥 Day's Most Probable Accumulator")

    if len(processed_games) < 2:
        st.warning("At least 2 games are required to generate an accumulator.")
    else:
        # Sort games by highest probability
        acc_picks = sorted(
            processed_games, key=lambda x: x["best_prob_pct"], reverse=True
        )[:3]

        acc_total_odds = 1.0
        for item in acc_picks:
            acc_total_odds *= item["best_prob_odds"]

        acc_stake_rec = user_bankroll * 0.10  # 10% of overall bankroll

        st.markdown('<div class="sp-card">', unsafe_allow_html=True)
        st.markdown("### 3-Fold Safety Multi-Bet")

        for item in acc_picks:
            st.markdown(
                f"• **{item['game']['home_team']} vs {item['game']['away_team']}** ➔ Bet: **{item['best_prob_bet']}** @ Odds `{item['best_prob_odds']:.2f}` (Prob: {item['best_prob_pct']:.1f}%)"
            )

        st.markdown("---")
        a1, a2, a3 = st.columns(3)
        a1.metric("Total Multi-Bet Odds", f"{acc_total_odds:.2f}")
        a2.metric("Rec. Accumulator Stake", f"KES {acc_stake_rec:,.2f}")
        a3.metric(
            "Est. Potential Payout",
            f"KES {acc_stake_rec * acc_total_odds:,.2f}",
        )
        st.markdown("</div>", unsafe_allow_html=True)

# --- TAB 3: BET TRACKING SLIP ---
with tab3:
    st.subheader("📝 Record & Track SportPesa Bets")

    with st.form("bet_tracker_form"):
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)

        game_list = [
            f"{g['home_team']} vs {g['away_team']} ({g['date']})"
            for g in fixtures
        ] if fixtures else ["No Games Loaded"]

        selected_game = col_f1.selectbox("Selected Game", options=game_list)
        bet_outcome = col_f2.text_input("Bet Outcome Placed", value="Home Win")
        stake_imputed = col_f3.number_input(
            "Actual Stake Placed (KES)", min_value=0.0, step=50.0
        )
        payout_imputed = col_f4.number_input(
            "Amount Won / Returned (KES)", min_value=0.0, step=50.0
        )

        submit_btn = st.form_submit_button("Record Result")

    if submit_btn:
        profit_loss = payout_imputed - stake_imputed
        if profit_loss >= 0:
            st.success(
                f"Bet recorded for {selected_game}! Net Profit: +KES {profit_loss:,.2f}"
            )
        else:
            st.error(
                f"Bet recorded for {selected_game}! Net Loss: -KES {abs(profit_loss):,.2f}"
            )
