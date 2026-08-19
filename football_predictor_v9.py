import json
import os
from datetime import date, datetime
import numpy as np
import pandas as pd
import streamlit as st

# ==============================================================================
# UI CONFIGURATION & CUSTOM VISUAL STYLING
# ==============================================================================
st.set_page_config(
    page_title="V9.3 Interactive Betting Engine", layout="wide", page_icon="⚽"
)

st.markdown(
    """
<style>
    .stApp { background-color: #0d1117; color: #f0f6fc; }
    
    .bet-card {
        background: linear-gradient(135deg, #161b22 0%, #21262d 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.4);
    }
    
    .history-card {
        background: #161b22;
        border-left: 5px solid #30363d;
        border-top: 1px solid #30363d;
        border-right: 1px solid #30363d;
        border-bottom: 1px solid #30363d;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    
    .badge-win { background-color: #238636; color: white; padding: 4px 10px; border-radius: 12px; font-weight: bold; font-size: 0.8rem; }
    .badge-loss { background-color: #da3633; color: white; padding: 4px 10px; border-radius: 12px; font-weight: bold; font-size: 0.8rem; }
    .badge-pending { background-color: #d29922; color: black; padding: 4px 10px; border-radius: 12px; font-weight: bold; font-size: 0.8rem; }
    .badge-push { background-color: #8b949e; color: white; padding: 4px 10px; border-radius: 12px; font-weight: bold; font-size: 0.8rem; }
    
    .match-header {
        font-size: 0.95rem;
        font-weight: 700;
        color: #58a6ff;
        border-bottom: 1px solid #30363d;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }
    
    .team-box { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
    .team-logo { width: 36px; height: 36px; object-fit: contain; }
    .team-name { font-size: 1.3rem; font-weight: 800; color: #ffffff; }
    .vs-divider { font-size: 0.85rem; font-weight: 700; color: #8b949e; margin: 2px 0 8px 48px; }
    
    .value-box {
        background-color: rgba(46, 160, 67, 0.15);
        border: 1px solid #2ea043;
        border-radius: 8px;
        padding: 12px;
        margin-top: 15px;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("⚽ Predictor, Stake Allocator & P/L Tracker")
st.caption(
    "Set bankroll allocations, analyze matches with team badges, and track real-time profit graphs."
)
st.divider()

# ==============================================================================
# FIXTURE DATABASE
# ==============================================================================
FIXTURE_DATABASE = [
    {
        "date": "2026-08-19",
        "league": "La Liga",
        "match": "Atlético Madrid vs Málaga CF",
        "home": "Atlético Madrid",
        "away": "Málaga CF",
        "home_logo": "https://upload.wikimedia.org/wikipedia/en/f/f4/Atletico_Madrid_2017_logo.svg",
        "away_logo": "https://upload.wikimedia.org/wikipedia/en/8/82/Malaga_cf.svg",
        "suggested_bet": "Atlético Madrid -1.25 AH",
        "prob": 0.72,
        "odds": 1.95,
        "closing_odds": 1.88,
    },
    {
        "date": "2026-08-20",
        "league": "Serie A",
        "match": "Inter Milan vs Parma",
        "home": "Inter Milan",
        "away": "Parma",
        "home_logo": "https://upload.wikimedia.org/wikipedia/commons/0/05/FC_Internazionale_Milano_2021.svg",
        "away_logo": "https://upload.wikimedia.org/wikipedia/en/a/a9/Parma_Calcio_1913_logo.svg",
        "suggested_bet": "Inter Milan Win & Over 1.5 Goals",
        "prob": 0.68,
        "odds": 1.65,
        "closing_odds": 1.60,
    },
    {
        "date": "2026-08-21",
        "league": "Premier League (EPL)",
        "match": "Arsenal vs Coventry City",
        "home": "Arsenal",
        "away": "Coventry City",
        "home_logo": "https://upload.wikimedia.org/wikipedia/en/5/53/Arsenal_FC.svg",
        "away_logo": "https://upload.wikimedia.org/wikipedia/en/9/94/Coventry_City_FC_logo.svg",
        "suggested_bet": "Arsenal -1.5 Asian Handicap",
        "prob": 0.785,
        "odds": 1.75,
        "closing_odds": 1.68,
    },
    {
        "date": "2026-08-21",
        "league": "Premier League (EPL)",
        "match": "Chelsea vs Fulham",
        "home": "Chelsea",
        "away": "Fulham",
        "home_logo": "https://upload.wikimedia.org/wikipedia/en/c/cc/Chelsea_FC.svg",
        "away_logo": "https://upload.wikimedia.org/wikipedia/en/e/eb/Fulham_FC_%28shield%29.svg",
        "suggested_bet": "Over 2.5 Total Goals",
        "prob": 0.58,
        "odds": 1.85,
        "closing_odds": 1.80,
    },
    {
        "date": "2026-08-22",
        "league": "Bundesliga",
        "match": "Bayern Munich vs Augsburg",
        "home": "Bayern Munich",
        "away": "Augsburg",
        "home_logo": "https://upload.wikimedia.org/wikipedia/commons/1/1b/FC_Bayern_M%C3%BCnchen_logo_%282017%29.svg",
        "away_logo": "https://upload.wikimedia.org/wikipedia/en/c/c5/FC_Augsburg_logo.svg",
        "suggested_bet": "Bayern Munich -2.0 AH",
        "prob": 0.70,
        "odds": 1.90,
        "closing_odds": 1.82,
    },
]


# ==============================================================================
# AUTOMATED CLV TRACKER CLASS
# ==============================================================================
class AutomatedCLVTracker:

  def __init__(self, log_file="clv_tracker_2026_27.json"):
    self.log_file = log_file
    self.entries = self._load_tracker()

  def _load_tracker(self):
    if os.path.exists(self.log_file):
      try:
        with open(self.log_file, "r") as f:
          data = json.load(f)
          for entry in data:
            if "actual_stake" not in entry:
              entry["actual_stake"] = entry.get("recommended_stake", 0.0)
            if "recommended_stake" not in entry:
              entry["recommended_stake"] = entry.get("actual_stake", 0.0)
          return data
      except Exception:
        return []
    return []

  def log_bet(
      self,
      league: str,
      match: str,
      bet_type: str,
      taken_odds: float,
      closing_odds: float,
      recommended_stake: float,
      actual_stake: float,
  ):
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
        "recommended_stake": float(recommended_stake),
        "actual_stake": float(actual_stake),
        "payout": 0.0,
        "profit_loss": 0.0,
        "result": "PENDING",
    }
    self.entries.append(entry)
    self._save()
    return clv_pct

  def update_result(
      self, bet_id: int, result: str, actual_stake: float = None
  ):
    for entry in self.entries:
      if entry["id"] == bet_id:
        if actual_stake is not None:
          entry["actual_stake"] = float(actual_stake)

        stake = entry.get("actual_stake", entry.get("recommended_stake", 0.0))
        odds = entry["taken_odds"]
        res = result.upper()

        if res == "WIN":
          entry["payout"] = round(stake * odds, 2)
          entry["profit_loss"] = round(stake * (odds - 1.0), 2)
        elif res == "LOSS":
          entry["payout"] = 0.0
          entry["profit_loss"] = round(-stake, 2)
        elif res == "PUSH":
          entry["payout"] = round(stake, 2)
          entry["profit_loss"] = 0.0

        entry["result"] = res
        self._save()
        return True
    return False

  def _save(self):
    with open(self.log_file, "w") as f:
      json.dump(self.entries, f, indent=4)


tracker = AutomatedCLVTracker()

# ==============================================================================
# DASHBOARD TABS
# ==============================================================================
tab1, tab2, tab3 = st.tabs([
    "📅 Match Selection & Staking Panel",
    "📈 Visual Performance Analytics",
    "📜 Visual Bet Slip History",
])

# ------------------------------------------------------------------------------
# TAB 1: MATCH SELECTION & STAKING PANEL
# ------------------------------------------------------------------------------
with tab1:
  st.subheader("1. Capital & Risk Configuration")

  col_bankroll, col_kelly = st.columns([1, 1])
  with col_bankroll:
    total_bankroll = st.number_input(
        "Total Bankroll Capital (100%)",
        value=10000.0,
        step=500.0,
        help="Enter your total available sports betting bankroll.",
    )
  with col_kelly:
    kelly_fraction = st.select_slider(
        "Model Risk Strategy (Kelly Fraction)",
        options=[
            "1/8 Kelly (Conservative)",
            "1/4 Kelly (Recommended)",
            "Half Kelly (Aggressive)",
        ],
        value="1/4 Kelly (Recommended)",
    )
    frac_mult = (
        0.125
        if "1/8" in kelly_fraction
        else (0.25 if "1/4" in kelly_fraction else 0.50)
    )

  st.divider()
  st.subheader("2. Fixture Selection & Visual Match Card")

  c_filter, c_card = st.columns([1, 1.2])

  with c_filter:
    selected_date = st.date_input(
        "Select Match Date",
        value=date(2026, 8, 21),
        min_value=date(2026, 8, 1),
        max_value=date(2026, 9, 30),
    )
    date_str = selected_date.strftime("%Y-%m-%d")

    matches_on_date = [m for m in FIXTURE_DATABASE if m["date"] == date_str]

    if matches_on_date:
      leagues = sorted(list(set([m["league"] for m in matches_on_date])))
      selected_league = st.selectbox("Select League", leagues)
      league_matches = [
          m for m in matches_on_date if m["league"] == selected_league
      ]
      selected_match_title = st.selectbox(
          "Select Match", [m["match"] for m in league_matches]
      )
      match_data = next(
          (m for m in league_matches if m["match"] == selected_match_title),
          None,
      )
    else:
      st.warning(
          f"No scheduled fixtures found for {date_str}. Showing fallback"
          " match."
      )
      match_data = FIXTURE_DATABASE[2]

  with c_card:
    if match_data:
      p = match_data["prob"]
      b = match_data["odds"] - 1.0
      q = 1.0 - p
      raw_kelly = (b * p - q) / b if b > 0 else 0
      rec_stake_pct = max(0.0, min(raw_kelly * frac_mult * 100.0, 3.0))
      rec_stake_cash = (rec_stake_pct / 100.0) * total_bankroll
      ev_pct = ((p * match_data["odds"]) - 1.0) * 100.0

      st.markdown(
          f"""
            <div class="bet-card">
                <div class="match-header">⚽ {match_data['league'].upper()} &nbsp;|&nbsp; {match_data['date']}</div>
                
                <div class="team-box">
                    <img src="{match_data['home_logo']}" class="team-logo" />
                    <span class="team-name">{match_data['home']}</span>
                </div>
                
                <div class="vs-divider">VS</div>
                
                <div class="team-box">
                    <img src="{match_data['away_logo']}" class="team-logo" />
                    <span class="team-name">{match_data['away']}</span>
                </div>
                
                <p style="color:#8b949e; margin-top:14px; margin-bottom:0;">
                    Model Win Prob: <b>{p*100:.1f}%</b> &nbsp;|&nbsp; Calculated Edge: <b style="color:#3fb950;">+{ev_pct:.2f}% EV</b>
                </p>
                
                <div class="value-box">
                    <span style="float:right; font-size:1.3rem; font-weight:bold; color:#2ea043;">{match_data['odds']}</span>
                    <div style="color:#3fb950; font-weight:bold; font-size:0.85rem; text-transform:uppercase;">🎯 RECOMMENDED VALUE SELECTION</div>
                    <div style="font-size:1.1rem; font-weight:bold; margin-top:4px;">{match_data['suggested_bet']}</div>
                </div>
            </div>
            """,
          unsafe_allow_html=True,
      )

      st.markdown("#### 3. Log Bet Slip Wager")
      with st.form("stake_entry_form"):
        st.info(
            f"💡 **Model Recommended Stake:** **{rec_stake_pct:.2f}%** of"
            f" Bankroll (**{rec_stake_cash:,.2f}** units)"
        )
        actual_stake_input = st.number_input(
            "Enter Actual Stake Wagered",
            value=float(round(rec_stake_cash, 2)),
            step=50.0,
        )

        log_submitted = st.form_submit_button("⚡ Place & Log Bet Slip")
        if log_submitted:
          clv = tracker.log_bet(
              match_data["league"],
              match_data["match"],
              match_data["suggested_bet"],
              match_data["odds"],
              match_data["closing_odds"],
              rec_stake_cash,
              actual_stake_input,
          )
          st.success(f"Bet Slip Logged! Calculated CLV: {clv:+.2f}%")

# ------------------------------------------------------------------------------
# TAB 2: VISUAL PERFORMANCE ANALYTICS
# ------------------------------------------------------------------------------
with tab2:
  st.subheader("Match Result Input Window")

  pending_bets = [e for e in tracker.entries if e.get("result") == "PENDING"]

  if pending_bets:
    c_update1, c_update2 = st.columns([1.5, 1])

    with c_update1:
      bet_options = {
          f"ID {b['id']}: {b['match']} ({b['bet_type']} @ {b['taken_odds']})": (
              b["id"]
          )
          for b in pending_bets
      }
      selected_bet_label = st.selectbox(
          "Select Pending Wager to Update", list(bet_options.keys())
      )
      target_id = bet_options[selected_bet_label]
      target_bet = next(b for b in pending_bets if b["id"] == target_id)

    with c_update2:
      with st.form("result_update_form"):
        default_stake_val = float(
            target_bet.get(
                "actual_stake", target_bet.get("recommended_stake", 0.0)
            )
        )
        actual_staked_final = st.number_input(
            "Final Staked Amount", value=default_stake_val
        )
        match_outcome = st.selectbox(
            "Match Outcome", ["WIN", "LOSS", "PUSH"]
        )
        update_btn = st.form_submit_button("Submit Match Result")

        if update_btn:
          tracker.update_result(
              target_id, match_outcome, actual_staked_final
          )
          st.success(f"Bet ID {target_id} updated to {match_outcome}!")
          st.rerun()
  else:
    st.info("No pending bets waiting for result input.")

  st.divider()
  st.subheader("📊 Visual Cumulative Profit & Bankroll Curve")

  settled_bets = [e for e in tracker.entries if e.get("result") != "PENDING"]

  if settled_bets:
    total_wagered = sum(
        b.get("actual_stake", b.get("recommended_stake", 0.0))
        for b in settled_bets
    )
    total_pnl = sum(b.get("profit_loss", 0.0) for b in settled_bets)
    wins = len([b for b in settled_bets if b.get("result") == "WIN"])
    losses = len([b for b in settled_bets if b.get("result") == "LOSS"])
    total_settled = wins + losses

    win_rate = (wins / total_settled * 100.0) if total_settled > 0 else 0.0
    roi_pct = (total_pnl / total_wagered * 100.0) if total_wagered > 0 else 0.0
    avg_clv = (
        np.mean([b.get("clv_pct", 0.0) for b in tracker.entries])
        if tracker.entries
        else 0.0
    )

    # Key Summary Metrics Cards
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Settled Wagers", f"{len(settled_bets)}")
    m2.metric("Total Capital Staked", f"{total_wagered:,.2f}")
    m3.metric("Net Profit / Loss", f"{total_pnl:+,.2f}")
    m4.metric("ROI Yield", f"{roi_pct:+.2f}%")
    m5.metric("Win Rate", f"{win_rate:.1f}%")

    # Interactive Profit Growth Chart
    chart_data = []
    running_pnl = 0.0
    for idx, bet in enumerate(settled_bets, start=1):
      running_pnl += bet.get("profit_loss", 0.0)
      chart_data.append(
          {"Bet Number": idx, "Cumulative Profit": running_pnl, "Date": bet["date"]}
      )

    df_chart = pd.DataFrame(chart_data)
    st.line_chart(df_chart, x="Bet Number", y="Cumulative Profit")

    # Visual Outcome Breakdown
    st.markdown("#### Wager Outcome Distribution")
    st.progress(win_rate / 100.0 if win_rate > 0 else 0.0)
    st.caption(
        f"🟢 **Wins:** {wins} &nbsp;|&nbsp; 🔴 **Losses:** {losses} &nbsp;|&nbsp;"
        f" 🟡 **Pending:** {len(pending_bets)}"
    )

  else:
    st.info("No settled bets recorded yet. Mark a pending wager to generate charts.")

# ------------------------------------------------------------------------------
# TAB 3: VISUAL BET SLIP HISTORY
# ------------------------------------------------------------------------------
with tab3:
  st.subheader("📜 Visual Bet Slip Cards")

  if tracker.entries:
    for bet in reversed(tracker.entries):
      res = bet.get("result", "PENDING").upper()
      if res == "WIN":
        badge_html = '<span class="badge-win">🟢 WIN</span>'
        pnl_text = f'<span style="color:#3fb950; font-weight:bold;">+{bet.get("profit_loss",0.0):,.2f}</span>'
      elif res == "LOSS":
        badge_html = '<span class="badge-loss">🔴 LOSS</span>'
        pnl_text = f'<span style="color:#f85149; font-weight:bold;">{bet.get("profit_loss",0.0):,.2f}</span>'
      elif res == "PUSH":
        badge_html = '<span class="badge-push">⚪ PUSH</span>'
        pnl_text = '<span>0.00</span>'
      else:
        badge_html = '<span class="badge-pending">🟡 PENDING</span>'
        pnl_text = '<span style="color:#8b949e;">Awaiting Result</span>'

      st.markdown(
          f"""
            <div class="history-card">
                <div style="display:flex; justify-between; align-items:center; margin-bottom:6px;">
                    <span style="color:#8b949e; font-size:0.85rem;"><b>ID #{bet['id']}</b> | {bet['date']} | {bet['league']}</span>
                    <div>{badge_html}</div>
                </div>
                <div style="font-size:1.15rem; font-weight:bold; color:#f0f6fc;">{bet['match']}</div>
                <div style="color:#58a6ff; font-weight:600; margin-top:4px;">Selection: {bet['bet_type']} @ {bet['taken_odds']}</div>
                <div style="display:flex; gap:20px; margin-top:10px; font-size:0.9rem; color:#8b949e; border-top:1px solid #21262d; padding-top:8px;">
                    <div>Stake: <b style="color:#ffffff;">{bet.get('actual_stake',0.0):,.2f}</b></div>
                    <div>CLV: <b style="color:#3fb950;">{bet.get('clv_pct',0.0):+.2f}%</b></div>
                    <div>Profit/Loss: {pnl_text}</div>
                </div>
            </div>
            """,
          unsafe_allow_html=True,
      )
  else:
    st.info("No bet slips logged yet.")
