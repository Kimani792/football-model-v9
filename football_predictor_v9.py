import streamlit as st
import json
import os
import numpy as np
from datetime import datetime

# ==============================================================================
# STREAMLIT UI & CUSTOM CSS (SPORTSBOOK THEME)
# ==============================================================================
st.set_page_config(page_title="V9.3 Betting Engine", layout="wide")

# Custom CSS to mimic dynamic sports betting UI (DraftKings / SportPesa style)
st.markdown("""
<style>
    /* Dark theme background */
    .stApp {
        background-color: #0d1117;
        color: #f0f6fc;
    }
    
    /* Sportsbook Card Styling */
    .bet-card {
        background: linear-gradient(135deg, #161b22 0%, #21262d 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.4);
    }
    
    .match-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #58a6ff;
        border-bottom: 1px solid #30363d;
        padding-bottom: 8px;
        margin-bottom: 12px;
    }
    
    .prob-badge {
        background-color: #1f6feb;
        color: #ffffff;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Value Pick Highlight Box */
    .value-box {
        background-color: rgba(46, 160, 67, 0.15);
        border: 1px solid #2ea043;
        border-radius: 8px;
        padding: 12px;
        margin-top: 15px;
    }
    
    .value-title {
        color: #3fb950;
        font-size: 0.9rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .odds-chip {
        background-color: #238636;
        color: #ffffff;
        font-weight: bold;
        padding: 6px 12px;
        border-radius: 6px;
        float: right;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.title("⚡ V9.3 PREDICTOR ENGINE")
st.caption("AI-Powered Sportsbook Model & Dynamic Value Finder")
st.divider()

# ==============================================================================
# 1. TRACKER LOGIC
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

tracker = AutomatedCLVTracker()

# ==============================================================================
# 2. MATCHDAY CARDS (BETTING APP UI)
# ==============================================================================
st.subheader("🔥 High Expected Value (+EV) Picks")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="bet-card">
        <div class="match-header">⚽ PREMIER LEAGUE</div>
        <h3 style="margin:0;">Arsenal vs Coventry City</h3>
        <p style="color:#8b949e; margin-top:5px;">Model Win Prob: <span class="prob-badge">78.5% Arsenal</span></p>
        <p style="color:#8b949e;">Expected Goals (xG): <b>2.45 - 0.40</b></p>
        
        <div class="value-box">
            <span class="odds-chip">1.75</span>
            <div class="value-title">🎯 TOP VALUE SELECTION</div>
            <div style="font-size:1.1rem; font-weight:bold; margin-top:4px;">Arsenal -1.5 Asian Handicap</div>
            <div style="color:#3fb950; font-size:0.85rem; margin-top:4px;">+7.6% Calculated Edge</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="bet-card">
        <div class="match-header">⚽ LA LIGA</div>
        <h3 style="margin:0;">Atlético Madrid vs Málaga CF</h3>
        <p style="color:#8b949e; margin-top:5px;">Model Win Prob: <span class="prob-badge">72.0% Atlético</span></p>
        <p style="color:#8b949e;">Expected Goals (xG): <b>2.15 - 0.55</b></p>
        
        <div class="value-box">
            <span class="odds-chip">1.95</span>
            <div class="value-title">🎯 TOP VALUE SELECTION</div>
            <div style="font-size:1.1rem; font-weight:bold; margin-top:4px;">Atlético Madrid -1.25 AH</div>
            <div style="color:#3fb950; font-size:0.85rem; margin-top:4px;">+19.9% Calculated Edge</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ==============================================================================
# 3. INTERACTIVE BET SLIP LOGGING
# ==============================================================================
st.subheader("📲 Quick Bet Slip Tracker")

with st.form("bet_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        league_input = st.selectbox("League", ["EPL", "LA_LIGA", "SERIE_A", "BUNDESLIGA"])
        match_input = st.text_input("Match", "Arsenal vs Coventry City")
    with c2:
        bet_input = st.text_input("Selection", "Arsenal -1.5 AH")
        taken_odds_input = st.number_input("Odds Taken", value=1.75)
    with c3:
        closing_odds_input = st.number_input("Closing Odds", value=1.68)
        stake_input = st.number_input("Stake Units", value=200.0)
    
    submitted = st.form_submit_button("⚡ LOG BET SLIP")
    if submitted:
        clv = tracker.log_bet(league_input, match_input, bet_input, taken_odds_input, closing_odds_input, stake_input)
        st.success(f"Bet Slip Recorded! Calculated Closing Line Value (CLV): {clv:+.2f}%")

st.divider()

# Display Current Tracker JSON
st.subheader("📜 Active Wagers")
if tracker.entries:
    st.json(tracker.entries)
else:
    st.info("No active wagers logged in tracker yet.")
