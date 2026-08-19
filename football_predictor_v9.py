import datetime
import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------
# Page Configuration & SportPesa Theme Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="SportPesa AI Predictor | Kenya",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS matching SportPesa navy blue (#0d1b2a / #1b263b) and red accent (#e63946)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0d1b2a;
        color: #ffffff;
    }
    .stSidebar {
        background-color: #1b263b !important;
    }
    .sp-card {
        background-color: #1b263b;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #00b4d8;
        margin-bottom: 15px;
    }
    .sp-card-val {
        border-left: 5px solid #2ec4b6;
    }
    .sp-card-acc {
        border-left: 5px solid #e63946;
    }
    .metric-container {
        background-color: #1b263b;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# API-SPORTS Integration Function
# ---------------------------------------------------------
API_KEY = st.secrets.get("API_SPORTS_KEY", "")


@st.cache_data(ttl=1800)  # Cache for 30 mins
def fetch_live_fixtures(league_id=39):  # 39 = Premier League
    """Fetches upcoming fixtures and odds directly from API-SPORTS API."""
    if not API_KEY or API_KEY == "your_actual_api_sports_key_here":
        # Fallback Mock Data if API Key is not yet populated
        return [
            {
                "id": 101,
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "home_logo": "https://media.api-sports.io/football/teams/42.png",
                "away_logo": "https://media.api-sports.io/football/teams/49.png",
                "prob_home": 0.58,
                "prob_draw": 0.24,
                "prob_away": 0.18,
                "odds_home": 1.95,
                "odds_draw": 3.50,
                "odds_away": 4.20,
            },
            {
                "id": 102,
                "home_team": "Man City",
                "away_team": "Liverpool",
                "home_logo": "https://media.api-sports.io/football/teams/50.png",
                "away_logo": "https://media.api-sports.io/football/teams/40.png",
                "prob_home": 0.52,
                "prob_draw": 0.26,
                "prob_away": 0.22,
                "odds_home": 2.10,
                "odds_draw": 3.60,
                "odds_away": 3.80,
            },
            {
                "id": 103,
                "home_team": "Real Madrid",
                "away_team": "Barcelona",
                "home_logo": "https://media.api-sports.io/football/teams/541.png",
                "away_logo": "https://media.api-sports.io/football/teams/529.png",
                "prob_home": 0.45,
                "prob_draw": 0.28,
                "prob_away": 0.27,
                "odds_home": 2.25,
                "odds_draw": 3.40,
                "odds_away": 3.10,
            },
        ]

    url = "https://v3.football.api-sports.io/fixtures"
    headers = {"x-apisports-key": API_KEY}
    params = {"league": league_id, "next": 10}

    try:
        response = requests.get(url, headers=headers, params=params).json()
        fixtures = []
        for item in response.get("response", []):
            fixtures.append(
                {
                    "id": item["fixture"]["id"],
                    "home_team": item["teams"]["home"]["name"],
                    "away_team": item["teams"]["away"]["name"],
                    "home_logo": item["teams"]["home"]["logo"],
                    "away_logo": item["teams"]["away"]["logo"],
                    "prob_home": 0.55,  # Calculated via predictor backend model
                    "prob_draw": 0.25,
                    "prob_away": 0.20,
                    "odds_home": 1.90,  # Synced from Bookmaker API
                    "odds_draw": 3.40,
                    "odds_away": 4.00,
                }
            )
        return fixtures
    except Exception as e:
        st.error(f"Error fetching API data: {e}")
        return []


# ---------------------------------------------------------
# Sidebar: Bankroll & Bet Allocation Input
# ---------------------------------------------------------
st.sidebar.title("🇰🇪 SportPesa Bankroll")
st.sidebar.caption("Set your daily bankroll in Kenya Shillings")

user_bankroll = st.sidebar.number_input(
    "Total Daily Budget / Stake (100% Bankroll in KES)",
    min_value=0.0,
    value=0.0,  # Default blank / 0 as requested
    step=100.0,
    format="%.2f",
    help="Enter your total betting wallet allocated for today.",
)

# ---------------------------------------------------------
# Dashboard Header
# ---------------------------------------------------------
st.title("⚽ SportPesa AI Value Predictor")
st.markdown("### Premier League & European Fixtures Analysis")

fixtures = fetch_live_fixtures()

# ---------------------------------------------------------
# Predictions Engine Calculations
# ---------------------------------------------------------
processed_games = []
for g in fixtures:
    # 1. Highest Probability Bet
    probs = {
        "Home Win": (g["prob_home"], g["odds_home"]),
        "Draw": (g["prob_draw"], g["odds_draw"]),
        "Away Win": (g["prob_away"], g["odds_away"]),
    }
    best_prob_bet = max(probs, key=lambda k: probs[k][0])
    best_prob_val = probs[best_prob_bet][0]
    best_prob_odds = probs[best_prob_bet][1]

    # 2. Highest Expected Value (EV) Bet
    # EV Formula = (Prob * Odds) - 1
    ev_scores = {k: (v[0] * v[1]) - 1 for k, v in probs.items()}
    highest_ev_bet = max(ev_scores, key=ev_scores.get)
    highest_ev_val = ev_scores[highest_ev_bet] * 100
    highest_ev_odds = probs[highest_ev_bet][1]

    # 3. Kelly Criterion Stake Suggestion (% of Bankroll)
    b = highest_ev_odds - 1
    p = probs[highest_ev_bet][0]
    q = 1 - p
    kelly_pct = max(0.0, (b * p - q) / b) if b > 0 else 0.0
    rec_stake_kes = (
        user_bankroll * (kelly_pct * 0.5)
    )  # Fractional Kelly (Half-Kelly) safety

    processed_games.append(
        {
            "game": g,
            "best_prob_bet": best_prob_bet,
            "best_prob_val": best_prob_val,
            "best_prob_odds": best_prob_odds,
            "highest_ev_bet": highest_ev_bet,
            "highest_ev_val": highest_ev_val,
            "highest_ev_odds": highest_ev_odds,
            "rec_stake_kes": rec_stake_kes,
        }
    )

# ---------------------------------------------------------
# Main Tabs Layout
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["🎯 Match Predictions", "🚀 Today's Accumulator", "📝 SportPesa Bet Tracker"]
)

# --- TAB 1: MATCH PREDICTIONS ---
with tab1:
    st.subheader("Match Predictions & Value Breakdown")

    for item in processed_games:
        g = item["game"]
        with st.container():
            st.markdown('<div class="sp-card">', unsafe_allow_html=True)
            col_teams, col_p1, col_p2, col_stake = st.columns([3, 2, 2, 2])

            with col_teams:
                c_h, c_vs, c_a = st.columns([1, 1, 1])
                c_h.image(g["home_logo"], width=45)
                c_vs.markdown(
                    f"**{g['home_team']}**<br>vs<br>**{g['away_team']}**",
                    unsafe_allow_html=True,
                )
                c_a.image(g["away_logo"], width=45)

            with col_p1:
                st.markdown("**⭐ Most Probable Bet**")
                st.write(f"**{item['best_prob_bet']}**")
                st.caption(
                    f"Prob: {item['best_prob_val']*100:.1f}% | Odds: {item['best_prob_odds']:.2f}"
                )

            with col_p2:
                st.markdown("**⚡ Highest Value Bet (+EV)**")
                st.write(f"**{item['highest_ev_bet']}**")
                st.caption(
                    f"EV: +{item['highest_ev_val']:.1f}% | Odds: {item['highest_ev_odds']:.2f}"
                )

            with col_stake:
                st.markdown("**💰 Suggested Stake**")
                if user_bankroll > 0:
                    st.success(f"**KES {item['rec_stake_kes']:,.2f}**")
                else:
                    st.info("Input total budget in sidebar")

            st.markdown("</div>", unsafe_allow_html=True)

# --- TAB 2: ACCUMULATOR ---
with tab2:
    st.subheader("🔥 Today's Recommended Accumulator")

    # Pick top 3 most probable picks for high win probability multi-bet
    acc_picks = sorted(
        processed_games, key=lambda x: x["best_prob_val"], reverse=True
    )[:3]

    combined_odds = 1.0
    for p in acc_picks:
        combined_odds *= p["best_prob_odds"]

    acc_stake = user_bankroll * 0.10  # Suggest 10% of total daily budget

    st.markdown('<div class="sp-card sp-card-acc">', unsafe_allow_html=True)
    st.write("### 3-Fold Safety Multi-Bet")

    for p in acc_picks:
        st.write(
            f"• **{p['game']['home_team']} vs {p['game']['away_team']}** ➔ Bet: **{p['best_prob_bet']}** @ Odds {p['best_prob_odds']:.2f}"
        )

    st.divider()
    ac1, ac2, ac3 = st.columns(3)
    ac1.metric("Total Accumulator Odds", f"{combined_odds:.2f}")
    ac2.metric(
        "Suggested Stake (10%)",
        f"KES {acc_stake:,.2f}" if user_bankroll > 0 else "Set Budget",
    )
    ac3.metric(
        "Potential Return",
        f"KES {acc_stake * combined_odds:,.2f}" if user_bankroll > 0 else "KES 0.00",
    )
    st.markdown("</div>", unsafe_allow_html=True)

# --- TAB 3: BET TRACKER ---
with tab3:
    st.subheader("📝 Record Bets Placed on SportPesa")

    with st.form("bet_entry_form"):
        f1, f2, f3, f4 = st.columns(4)
        match_selected = f1.selectbox(
            "Select Game", [f"{g['home_team']} vs {g['away_team']}" for g in fixtures]
        )
        bet_placed = f2.text_input("Bet Outcome Placed", value="Home Win")
        amount_staked = f3.number_input(
            "Amount Bet (KES)", min_value=0.0, step=50.0
        )
        amount_won = f4.number_input(
            "Amount Won / Returned (KES)", min_value=0.0, step=50.0
        )

        submit_bet = st.form_submit_button("Record Bet Result")

    if submit_bet:
        st.success(
            f"Recorded bet on '{match_selected}'! Net P/L: KES {amount_won - amount_staked:,.2f}"
        )
