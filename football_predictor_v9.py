import streamlit as st
import json
import os
import numpy as np
from datetime import datetime

# ==============================================================================
# STREAMLIT UI CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="V9.3 Predictor Engine", layout="wide")

st.title("⚽ Football Predictor Engine V9.3")
st.caption("Multi-League Probability & Value Tracker")

# ==============================================================================
# 1. MULTI-LEAGUE CLV TRACKER
# ==============================================================================
class AutomatedCLVTracker:
    def __init__(self, log_file="clv_tracker_2026_27.json"):
        self.log_file = log_file
        self.entries = self._load_tracker()

    def _load_tracker(self):
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, "r") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def log_bet(self, league: str, match: str, bet_type: str, taken_odds: float, closing_odds: float, stake: float):
        clv_pct = ((taken_odds / closing_odds) - 1.0) * 100.0
        entry = {
            "id": len(self.entries) + 1,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "league": league.upper(),
            "match": match,
            "bet_type": bet_type,
            "taken_odds": float(taken_odds),
            "closing_odds": float(closing_odds),
            "clv_pct": round(clv_pct, 2),
            "stake": float(stake),
            "result": "PENDING"
        }
        self.entries.append(entry)
        with open(self.log_file, "w") as f:
            json.dump(self.entries, f, indent=4)
        return clv_pct

# Initialize Tracker
tracker = AutomatedCLVTracker()

# ==============================================================================
# 2. DASHBOARD DISPLAY
# ==============================================================================
st.subheader("📊 Matchday Model Selections")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Arsenal vs Coventry City")
    st.write("**League:** Premier League (EPL)")
    st.write("**Model Projection:** Arsenal Win (78.5%)")
    st.success("🎯 **Best Value:** Arsenal -1.5 Asian Handicap @ 1.75 (+7.6% EV)")

with col2:
    st.markdown("### Atlético Madrid vs Málaga CF")
    st.write("**League:** La Liga")
    st.write("**Model Projection:** Atlético Win (72.0%)")
    st.success("🎯 **Best Value:** Atlético Madrid -1.25 AH @ 1.95 (+19.9% EV)")

st.divider()

# Interactive Bet Logging
st.subheader("📝 Log New Bet to Tracker")
with st.form("bet_form"):
    league_input = st.selectbox("League", ["EPL", "LA_LIGA", "SERIE_A", "BUNDESLIGA"])
    match_input = st.text_input("Match Name", "Arsenal vs Coventry City")
    bet_input = st.text_input("Selection", "Arsenal -1.5 AH")
    taken_odds_input = st.number_input("Odds Taken", value=1.75)
    closing_odds_input = st.number_input("Closing Odds", value=1.68)
    stake_input = st.number_input("Stake Units", value=200.0)
    
    submitted = st.form_submit_button("Log Bet")
    if submitted:
        clv = tracker.log_bet(league_input, match_input, bet_input, taken_odds_input, closing_odds_input, stake_input)
        st.success(f"Bet Logged Successfully! Calculated CLV: {clv:+.2f}%")

st.divider()

# Display Current Logged Bets
st.subheader("📜 Recent Logged Bets")
if tracker.entries:
    st.json(tracker.entries)
else:
    st.info("No bets logged in `clv_tracker_2026_27.json` yet.")
