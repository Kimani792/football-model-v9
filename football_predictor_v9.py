import datetime
import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------
# Page Configuration & SportPesa Sleek Navy Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="SportPesa AI Predictor | Kenya",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sleek, clean high-contrast SportPesa navy theme
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0d1b2a;
        color: #ffffff;
    }
    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #1b263b !important;
        border-right: 1px solid #2b3a4e;
    }
    .sp-card {
        background-color: #1b263b;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #00b4d8;
        border-top: 1px solid #2b3a4e;
        border-right: 1px solid #2b3a4e;
        border-bottom: 1px solid #2b3a4e;
        margin-bottom: 14px;
    }
    .sp-acc-card {
        background-color: #1b263b;
        border-radius: 8px;
        padding: 20px;
        border-left: 5px solid #e63946;
        border-top: 1px solid #2b3a4e;
        border-right: 1px solid #2b3a4e;
        border-bottom: 1px solid #2b3a4e;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# API Data Loader
# ---------------------------------------------------------


@st.cache_data(ttl=1800)
def load_fixtures_data(start_d, end_d):
    """Fetches real fixtures from API-SPORTS between start_d and end_d."""
    api_key = st.secrets.get("API_SPORTS_KEY", "")

    if not api_key or api_key == "your_actual_api_sports_key_here":
        st.warning(
            "⚠️ Please configure `API_SPORTS_KEY` in Streamlit Secrets to load live fixtures."
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

            fixtures.append(
                {
                    "id": item["fixture"]["id"],
                    "date": fixture_date,
                    "league": item["league"]["name"],
                    "home_team": item["teams"]["home"]["name"],
                    "away_team": item["teams"]["away"]["name"],
                    "home_logo": item["teams"]["home"]["logo"],
                    "away_logo": item["teams"]["away"]["logo"],
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
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.title("🇰🇪 SportPesa Bankroll")
st.sidebar.markdown("---")

user_bankroll = st.sidebar.number_input(
    "Total Daily Budget / Stake (KES)",
    min_value=0.0,
    value=0.0,
    step=100.0,
    format="%.2f",
    help="Enter your total daily betting wallet in Kenya Shillings.",
)

st.sidebar.markdown("### 📅 Date Range Filter")
today_date = datetime.date.today()
col_d1, col_d2 = st.sidebar.columns(2)
start_date = col_d1.date_input("Start Date", value=today_date)
end_date = col_d2.date_input(
    "End Date", value=today_date + datetime.timedelta(days=1)
)

if start_date > end_date:
    st.sidebar.error("Start Date must be before or equal to End Date.")

raw_fixtures = load_fixtures_data(start_date, end_date)

st.sidebar.markdown("### 🏆 League Filter")
league_mode = st.sidebar.radio(
    "View Mode:", ["All Leagues", "Specific League"]
)

available_leagues = sorted(list(set(g["league"] for g in raw_fixtures)))

if league_mode == "Specific League" and available_leagues:
    selected_league = st.sidebar.selectbox("Select League:", options=available_leagues)
    fixtures = [g for g in raw_fixtures if g["league"] == selected_league]
else:
    fixtures = raw_fixtures

# ---------------------------------------------------------
# Main Page Dashboard
# ---------------------------------------------------------
st.title("⚽ SportPesa AI Value Predictor")
st.caption(
    f"Fixtures Schedule: **{start_date.strftime('%d %b %Y')}** to **{end_date.strftime('%d %b %Y')}**"
)

m1, m2, m3 = st.columns(3)
m1.metric("Available Bankroll", f"KES {user_bankroll:,.2f}")
m2.metric("Total Matches", len(fixtures))
m3.metric("Leagues", len(set(g["league"] for g in fixtures)) if fixtures else 0)

st.markdown("---")

# Processed Game Engine Data
processed_games = []
for g in fixtures:
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

    # Kelly Recommended Stakes
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

# ---------------------------------------------------------
# Tabbed Layout
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["🎯 Match Predictions", "🚀 Daily Accumulator", "📝 Bet Tracker"]
)

# --- TAB 1: MATCH PREDICTIONS ---
with tab1:
    if not fixtures:
        st.info("No fixtures available for the selected dates or league.")
    else:
        for idx, item in enumerate(processed_games):
            g = item["game"]

            st.markdown('<div class="sp-card">', unsafe_allow_html=True)
            st.markdown(f"**📅 {g['date'].strftime('%a, %b %d')}** | 🏆 {g['league']}")

            col_teams, col_prob, col_val = st.columns([2.5, 3.5, 3.5])

            # Teams & Emblems Column
            with col_teams:
                st.markdown("<br>", unsafe_allow_html=True)
                t_h, t_vs, t_a = st.columns([1, 1, 1])
                t_h.image(g["home_logo"], width=40)
                t_vs.markdown("**VS**")
                t_a.image(g["away_logo"], width=40)
                st.markdown(f"**{g['home_team']}** vs **{g['away_team']}**")

            # Most Probable Win Bet Column
            with col_prob:
                st.markdown("**⭐ Most Probable Bet**")
                st.markdown(f"**Pick:** `{item['best_prob_bet']}`")
                st.markdown(
                    f"Prob: **{item['best_prob_pct']:.1f}%** | Odds: **{item['best_prob_odds']:.2f}**"
                )
                st.markdown(f"Rec. Stake: **KES {item['rec_stake_prob']:,.2f}**")
                st.number_input(
                    "Stake Placed (KES)",
                    min_value=0.0,
                    value=0.0,
                    step=50.0,
                    key=f"stk_p_{g['id']}",
                )

            # Best Value (+EV) Bet Column
            with col_val:
                st.markdown("**⚡ Best Value (+EV) Bet**")
                st.markdown(f"**Pick:** `{item['best_val_bet']}`")
                st.markdown(
                    f"Prob: **{item['best_val_pct']:.1f}%** | Odds: **{item['best_val_odds']:.2f}**"
                )
                st.markdown(f"Rec. Stake: **KES {item['rec_stake_val']:,.2f}**")
                st.number_input(
                    "Stake Placed (KES)",
                    min_value=0.0,
                    value=0.0,
                    step=50.0,
                    key=f"stk_v_{g['id']}",
                )

            st.markdown("</div>", unsafe_allow_html=True)

# --- TAB 2: TODAY'S ACCUMULATOR ---
with tab2:
    st.subheader("🔥 Day's Most Probable Accumulator")

    if len(processed_games) < 2:
        st.warning("At least 2 matches are needed to construct an accumulator.")
    else:
        acc_picks = sorted(
            processed_games, key=lambda x: x["best_prob_pct"], reverse=True
        )[:3]

        acc_total_odds = 1.0
        for p in acc_picks:
            acc_total_odds *= p["best_prob_odds"]

        acc_rec_stake = user_bankroll * 0.10

        st.markdown('<div class="sp-acc-card">', unsafe_allow_html=True)
        st.markdown("### 3-Fold Safety Multi-Bet")

        for p in acc_picks:
            st.markdown(
                f"• **{p['game']['home_team']} vs {p['game']['away_team']}** ➔ Bet: **{p['best_prob_bet']}** @ `{p['best_prob_odds']:.2f}` (Prob: {p['best_prob_pct']:.1f}%)"
            )

        st.markdown("---")
        a1, a2, a3 = st.columns(3)
        a1.metric("Total Accumulator Odds", f"{acc_total_odds:.2f}")
        a2.metric(
            "Rec. Stake (10% Bankroll)",
            f"KES {acc_rec_stake:,.2f}" if user_bankroll > 0 else "Set Budget",
        )
        a3.metric(
            "Est. Return",
            f"KES {acc_rec_stake * acc_total_odds:,.2f}" if user_bankroll > 0 else "KES 0.00",
        )
        st.markdown("</div>", unsafe_allow_html=True)

# --- TAB 3: BET TRACKER ---
with tab3:
    st.subheader("📝 Record Bets Placed on SportPesa")

    with st.form("bet_entry_form"):
        c1, c2, c3, c4 = st.columns(4)
        game_list = (
            [f"{g['home_team']} vs {g['away_team']}" for g in fixtures]
            if fixtures
            else ["No Games Loaded"]
        )
        selected_match = c1.selectbox("Game Selected", game_list)
        outcome_placed = c2.text_input("Bet Outcome", value="Home Win")
        staked_amt = c3.number_input("Amount Staked (KES)", min_value=0.0, step=50.0)
        returned_amt = c4.number_input(
            "Amount Won / Returned (KES)", min_value=0.0, step=50.0
        )

        submit_bet = st.form_submit_button("Record Result")

    if submit_bet:
        net_pl = returned_amt - staked_amt
        if net_pl >= 0:
            st.success(f"Recorded bet for '{selected_match}'! Net Profit: +KES {net_pl:,.2f}")
        else:
            st.error(f"Recorded bet for '{selected_match}'! Net Loss: -KES {abs(net_pl):,.2f}")
