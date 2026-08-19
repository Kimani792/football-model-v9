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

# High-contrast CSS styling
st.markdown(
    """
    <style>
    /* Global App Styling */
    .stApp {
        background-color: #0b131e;
        color: #ffffff;
    }
    
    /* High contrast text everywhere */
    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: #ffffff !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #111d2c !important;
        border-right: 1px solid #22354d;
    }
    
    /* Match Card Container */
    .sp-card {
        background-color: #162436;
        border-radius: 10px;
        padding: 18px;
        border: 1px solid #2a3f5a;
        margin-bottom: 20px;
    }
    
    /* Input Box High-Contrast Styling */
    div[data-baseweb="input"] {
        background-color: #0b131e !important;
        border-color: #00b4d8 !important;
        color: #ffffff !important;
    }
    
    /* Badges */
    .badge-prob {
        background-color: #00b4d8;
        color: #000000 !important;
        font-weight: bold;
        padding: 3px 8px;
        border-radius: 4px;
    }
    .badge-val {
        background-color: #FFD166;
        color: #000000 !important;
        font-weight: bold;
        padding: 3px 8px;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# API-SPORTS Data Loader / Fallback Mock Engine
# ---------------------------------------------------------
API_KEY = st.secrets.get("API_SPORTS_KEY", "")


@st.cache_data(ttl=1800)
def load_fixtures_data(start_d, end_d):
    """Fetches upcoming fixtures between start_date and end_date."""
    # Mock data populated with dates relative to today for demo
    today = datetime.date.today()
    tomorrow = today + datetime.timedelta(days=1)
    day_after = today + datetime.timedelta(days=2)

    all_data = [
        {
            "id": 101,
            "date": today,
            "league": "English Premier League",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "home_logo": "https://media.api-sports.io/football/teams/42.png",
            "away_logo": "https://media.api-sports.io/football/teams/49.png",
            "prob_home": 0.62,
            "prob_draw": 0.22,
            "prob_away": 0.16,
            "odds_home": 1.85,
            "odds_draw": 3.60,
            "odds_away": 4.50,
        },
        {
            "id": 102,
            "date": today,
            "league": "Spanish La Liga",
            "home_team": "Real Madrid",
            "away_team": "Barcelona",
            "home_logo": "https://media.api-sports.io/football/teams/541.png",
            "away_logo": "https://media.api-sports.io/football/teams/529.png",
            "prob_home": 0.48,
            "prob_draw": 0.27,
            "prob_away": 0.25,
            "odds_home": 2.20,
            "odds_draw": 3.40,
            "odds_away": 3.10,
        },
        {
            "id": 103,
            "date": tomorrow,
            "league": "German Bundesliga",
            "home_team": "Bayern Munich",
            "away_team": "Dortmund",
            "home_logo": "https://media.api-sports.io/football/teams/157.png",
            "away_logo": "https://media.api-sports.io/football/teams/165.png",
            "prob_home": 0.68,
            "prob_draw": 0.18,
            "prob_away": 0.14,
            "odds_home": 1.55,
            "odds_draw": 4.20,
            "odds_away": 5.50,
        },
        {
            "id": 104,
            "date": day_after,
            "league": "Italian Serie A",
            "home_team": "Inter Milan",
            "away_team": "AC Milan",
            "home_logo": "https://media.api-sports.io/football/teams/505.png",
            "away_logo": "https://media.api-sports.io/football/teams/489.png",
            "prob_home": 0.42,
            "prob_draw": 0.31,
            "prob_away": 0.27,
            "odds_home": 2.30,
            "odds_draw": 3.30,
            "odds_away": 3.20,
        },
    ]

    # Filter by date range
    filtered = [g for g in all_data if start_d <= g["date"] <= end_d]
    return filtered


# ---------------------------------------------------------
# Sidebar Inputs & Controls
# ---------------------------------------------------------
st.sidebar.title("🇰🇪 SportPesa Control Panel")
st.sidebar.markdown("---")

# 1. Total Stake Budget Input
user_bankroll = st.sidebar.number_input(
    "Total Daily Budget / Stake (KES)",
    min_value=0.0,
    value=1000.0,
    step=100.0,
    format="%.2f",
    help="Enter your available bankroll in Kenya Shillings.",
)

st.sidebar.markdown("### 📅 Date Filter")
# 2. Date Range Input (Start and End Date)
today_date = datetime.date.today()
col_d1, col_d2 = st.sidebar.columns(2)
start_date = col_d1.date_input("Bet Start Date", value=today_date)
end_date = col_d2.date_input(
    "Bet End Date", value=today_date + datetime.timedelta(days=2)
)

if start_date > end_date:
    st.sidebar.error("Error: Start Date must be before or equal to End Date.")

# Fetch data for selected date range
raw_fixtures = load_fixtures_data(start_date, end_date)

# 3. Dynamic League Selection
available_leagues = sorted(list(set(g["league"] for g in raw_fixtures)))
st.sidebar.markdown("### 🏆 League Filter")
league_mode = st.sidebar.radio(
    "Select League Mode:", ["All Leagues for Selected Date(s)", "Specific League"]
)

if league_mode == "Specific League" and available_leagues:
    selected_league = st.sidebar.selectbox(
        "Choose League:", options=available_leagues
    )
    fixtures = [g for g in raw_fixtures if g["league"] == selected_league]
else:
    fixtures = raw_fixtures

# ---------------------------------------------------------
# Main Page Header & Top Metrics
# ---------------------------------------------------------
st.title("⚽ SportPesa AI Match Predictor")
st.markdown(
    f"**Showing fixtures from {start_date.strftime('%b %d, %Y')} to {end_date.strftime('%b %d, %Y')}**"
)

m1, m2, m3 = st.columns(3)
m1.metric("Available Bankroll", f"KES {user_bankroll:,.2f}")
m2.metric("Fixtures Found", len(fixtures))
m3.metric(
    "Leagues Playing",
    (
        len(set(g["league"] for g in fixtures))
        if fixtures
        else 0
    ),
)

st.markdown("---")

# ---------------------------------------------------------
# Fixture List with High Contrast & Detailed Bet Columns
# ---------------------------------------------------------
if not fixtures:
    st.warning(
        "No fixtures found for the selected date range and league criteria."
    )
else:
    for idx, g in enumerate(fixtures):
        # Calculation logic for bets
        probs = {
            "Home Win": (g["prob_home"], g["odds_home"]),
            "Draw": (g["prob_draw"], g["odds_draw"]),
            "Away Win": (g["prob_away"], g["odds_away"]),
        }

        # 1. Most Probable Bet
        best_prob_outcome = max(probs, key=lambda k: probs[k][0])
        best_prob_pct = probs[best_prob_outcome][0] * 100
        best_prob_odds = probs[best_prob_outcome][1]

        # 2. Best Value Bet (+EV)
        ev_scores = {k: (v[0] * v[1]) - 1 for k, v in probs.items()}
        best_val_outcome = max(ev_scores, key=ev_scores.get)
        best_val_pct = probs[best_val_outcome][0] * 100
        best_val_odds = probs[best_val_outcome][1]

        # Kelly Stake Calculations
        b_prob = best_prob_odds - 1
        p_prob = probs[best_prob_outcome][0]
        kelly_prob = (
            max(0.0, (b_prob * p_prob - (1 - p_prob)) / b_prob)
            if b_prob > 0
            else 0.0
        )
        rec_stake_prob = user_bankroll * (kelly_prob * 0.25)  # 25% Kelly safety

        b_val = best_val_odds - 1
        p_val = probs[best_val_outcome][0]
        kelly_val = (
            max(0.0, (b_val * p_val - (1 - p_val)) / b_val) if b_val > 0 else 0.0
        )
        rec_stake_val = user_bankroll * (kelly_val * 0.25)

        # Rendering High-Contrast Game Card
        st.markdown('<div class="sp-card">', unsafe_allow_html=True)

        # Match Header Details
        st.markdown(
            f"#### 📅 {g['date'].strftime('%a, %b %d')} &nbsp;|&nbsp; 🏆 {g['league']}"
        )

        c_match, c_prob_bet, c_val_bet = st.columns([2.5, 3.5, 3.5])

        # Team Logos and Names
        with c_match:
            st.markdown("<br>", unsafe_allow_html=True)
            t1, t_vs, t2 = st.columns([1, 1, 1])
            t1.image(g["home_logo"], width=50)
            t_vs.markdown("### VS")
            t2.image(g["away_logo"], width=50)
            st.markdown(f"**{g['home_team']}** vs **{g['away_team']}**")

        # Column 1: Most Probable Bet Details
        with c_prob_bet:
            st.markdown(
                '<span class="badge-prob">MOST PROBABLE WIN BET</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Pick:** `{best_prob_outcome}`")
            st.markdown(
                f"• **Win Probability:** `{best_prob_pct:.1f}%`\n"
                f"• **Betting Odds:** `{best_prob_odds:.2f}`\n"
                f"• **Rec. Stake:** `KES {rec_stake_prob:,.2f}`"
            )
            st.number_input(
                f"Stake Placed (KES) - Probable #{idx+1}",
                min_value=0.0,
                value=0.0,
                step=50.0,
                key=f"stake_prob_{g['id']}",
            )

        # Column 2: Best Value Bet Details
        with c_val_bet:
            st.markdown(
                '<span class="badge-val">BEST VALUE (+EV) BET</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Pick:** `{best_val_outcome}`")
            st.markdown(
                f"• **Win Probability:** `{best_val_pct:.1f}%`\n"
                f"• **Betting Odds:** `{best_val_odds:.2f}`\n"
                f"• **Rec. Stake:** `KES {rec_stake_val:,.2f}`"
            )
            st.number_input(
                f"Stake Placed (KES) - Value #{idx+1}",
                min_value=0.0,
                value=0.0,
                step=50.0,
                key=f"stake_val_{g['id']}",
            )

        st.markdown("</div>", unsafe_allow_html=True)
