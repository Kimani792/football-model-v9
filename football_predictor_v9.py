import numpy as np
import pandas as pd
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Football Predictor & Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling to enforce clean layout
st.markdown(
    """
    <style>
    .stApp {
        padding-top: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #e9ecef;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Dummy Data Generators (Replace with API-SPORTS Data)
# ---------------------------------------------------------
@st.cache_data
def load_fixture_data():
    data = [
        {
            "Match": "Arsenal vs Chelsea",
            "League": "Premier League",
            "Home Win Prob": 0.58,
            "Draw Prob": 0.24,
            "Away Win Prob": 0.18,
            "Best Odds (Home)": 1.95,
            "Implied EV (%)": 13.1,
            "Recommended Bet": "Home Win",
            "Kelly Stake (%)": 3.2,
        },
        {
            "Match": "Real Madrid vs Barcelona",
            "League": "La Liga",
            "Home Win Prob": 0.45,
            "Draw Prob": 0.28,
            "Away Win Prob": 0.27,
            "Best Odds (Home)": 2.10,
            "Implied EV (%)": -5.5,
            "Recommended Bet": "No Value",
            "Kelly Stake (%)": 0.0,
        },
        {
            "Match": "Bayern Munich vs Dortmund",
            "League": "Bundesliga",
            "Home Win Prob": 0.65,
            "Draw Prob": 0.20,
            "Away Win Prob": 0.15,
            "Best Odds (Home)": 1.62,
            "Implied EV (%)": 5.3,
            "Recommended Bet": "Home Win",
            "Kelly Stake (%)": 1.8,
        },
        {
            "Match": "Inter Milan vs AC Milan",
            "League": "Serie A",
            "Home Win Prob": 0.38,
            "Draw Prob": 0.32,
            "Away Win Prob": 0.30,
            "Best Odds (Away)": 3.60,
            "Implied EV (%)": 8.0,
            "Recommended Bet": "Away Win",
            "Kelly Stake (%)": 2.1,
        },
    ]
    return pd.DataFrame(data)


@st.cache_data
def load_performance_history():
    dates = pd.date_range(end=pd.Timestamp.today(), periods=15, freq="D")
    np.random.seed(42)
    daily_returns = np.random.normal(loc=45, scale=120, size=15)
    cumulative_profit = np.cumsum(daily_returns) + 1000

    df = pd.DataFrame(
        {
            "Date": dates,
            "Daily P/L ($)": daily_returns,
            "Bankroll ($)": cumulative_profit,
            "CLV Beat Rate (%)": np.random.uniform(55, 75, size=15),
        }
    )
    return df


# Load Datasets
df_fixtures = load_fixture_data()
df_history = load_performance_history()

# ---------------------------------------------------------
# Sidebar Options
# ---------------------------------------------------------
st.sidebar.title("⚽ Betting Dashboard")
selected_league = st.sidebar.multiselect(
    "Filter by League",
    options=df_fixtures["League"].unique(),
    default=df_fixtures["League"].unique(),
)

min_ev = st.sidebar.slider(
    "Minimum Expected Value (EV %)", min_value=-10.0, max_value=20.0, value=0.0
)

# Filter Data
filtered_fixtures = df_fixtures[
    (df_fixtures["League"].isin(selected_league))
    & (df_fixtures["Implied EV (%)"] >= min_ev)
]

# ---------------------------------------------------------
# Header & Key Metrics
# ---------------------------------------------------------
st.title("Football Prediction Engine & Analytics")
st.caption("Live Value Bets, CLV Tracking, and Bankroll Management")

current_bankroll = df_history["Bankroll ($)"].iloc[-1]
total_profit = current_bankroll - 1000
clv_rate = df_history["CLV Beat Rate (%)"].mean()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Current Bankroll", f"${current_bankroll:,.2f}")
m2.metric(
    "Total Net Profit",
    f"${total_profit:,.2f}",
    delta=f"{total_profit:,.2f}",
    delta_color="normal",
)
m3.metric("Avg CLV Beat Rate", f"{clv_rate:.1f}%")
m4.metric("Active Value Bets", len(filtered_fixtures))

st.divider()

# ---------------------------------------------------------
# Main Content: Fixtures & Charts
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["🔥 Active Value Opportunities", "📈 Performance Analytics"])

with tab1:
    st.subheader("Upcoming Matches & Recommended Bets")

    if filtered_fixtures.empty:
        st.info("No matches meet the selected filter criteria.")
    else:
        for idx, row in filtered_fixtures.iterrows():
            with st.container():
                c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
                c1.markdown(f"**{row['Match']}**\n*{row['League']}*")
                c2.markdown(
                    f"**Probabilities:**\nH: {row['Home Win Prob']*100:.0f}% | D: {row['Draw Prob']*100:.0f}% | A: {row['Away Win Prob']*100:.0f}%"
                )

                ev_val = row["Implied EV (%)"]
                ev_color = "green" if ev_val > 0 else "red"
                c3.markdown(
                    f"**EV:** <span style='color:{ev_color}; font-weight:bold;'>{ev_val}%</span>",
                    unsafe_allow_html=True,
                )

                c4.markdown(
                    f"**Rec:** {row['Recommended Bet']}\n*(Stake: {row['Kelly Stake (%)']}%)*"
                )
                st.divider()

with tab2:
    st.subheader("Cumulative Profit Trajectory")
    st.line_chart(df_history, x="Date", y="Bankroll ($)")

    st.subheader("Historical Match Log")
    st.dataframe(df_history.tail(10), use_container_width=True)
