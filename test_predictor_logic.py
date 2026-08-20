"""
Tests for sportpesa_predictor_v2.py core logic — no network, no Streamlit
runtime needed. Run: python3 test_predictor_logic.py
"""
import json
import os
import sys
import types

# --- stub streamlit so the module under test can be imported headlessly ---
st_stub = types.ModuleType("streamlit")


class _Secrets(dict):
    def get(self, k, default=None):
        return dict.get(self, k, default)


st_stub.secrets = _Secrets()
st_stub.cache_data = lambda *a, **k: (lambda f: f)
st_stub.cache_resource = lambda *a, **k: (lambda f: f)
for name in ["set_page_config", "markdown", "title", "sidebar", "columns",
             "number_input", "date_input", "warning", "info", "error",
             "success", "metric", "tabs", "subheader", "form",
             "form_submit_button", "selectbox", "text_input", "image",
             "json", "radio", "caption"]:
    setattr(st_stub, name, lambda *a, **k: None)


class _FakeSidebar:
    def __getattr__(self, item):
        return lambda *a, **k: None


st_stub.sidebar = _FakeSidebar()
st_stub.session_state = {}
sys.modules["streamlit"] = st_stub

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import only the pure-logic pieces by executing the module in a controlled
# namespace up to (not including) the Streamlit UI calls -- simplest robust
# way here is to import the functions directly since they don't touch
# st.* at call time except via st.secrets/cache decorators already stubbed.
import importlib.util

spec = importlib.util.spec_from_file_location(
    "predictor", os.path.join(os.path.dirname(__file__), "sportpesa_predictor_v2.py")
)
predictor = importlib.util.module_from_spec(spec)

# The module runs Streamlit page calls at import time (st.set_page_config etc,
# all stubbed to no-ops above) and also reads st.secrets / sidebar widgets
# which are stubbed to return None -- fine, since we only test the pure
# functions below, not the page render.
try:
    spec.loader.exec_module(predictor)
except Exception as e:
    print(f"FAILED to import module under test: {e}")
    raise

passed = 0
failed = 0


def check(label, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ✓ {label}")
    else:
        failed += 1
        print(f"  ✗ {label}")


# ---------------------------------------------------------------
print("current_season()")
import datetime
season = predictor.current_season()
today = datetime.date.today()
expected = today.year if today.month >= 7 else today.year - 1
check("returns current season start-year given today's date", season == expected)

# ---------------------------------------------------------------
print("\ndevig_probabilities()")
result = predictor.devig_probabilities(1.95, 3.40, 4.10)
check("returns a 3-tuple", result is not None and len(result) == 3)
check("probabilities sum to 1.0", abs(sum(result) - 1.0) < 1e-9)
check("home is most likely given lowest odds", result[0] > result[1] > result[2])

result_missing = predictor.devig_probabilities(None, 3.40, 4.10)
check("returns None when odds incomplete (no fabrication)", result_missing is None)

# ---------------------------------------------------------------
print("\ndixon_coles_probabilities()")
ratings = {
    "La Liga": {
        "Real Madrid": {"attack": 1.55, "defense": 0.75},
        "Real Sociedad": {"attack": 1.05, "defense": 1.05},
    }
}
artifacts = {
    "params": {"dixon_coles": {"rho_domestic": -0.04043}},
    "home_adv": {"La Liga": {"new_ha": 1.2492}},
    "calibration": None,
}
probs = predictor.dixon_coles_probabilities("Real Madrid", "Real Sociedad", "La Liga", ratings, artifacts)
check("returns probabilities when both teams have ratings", probs is not None)
if probs:
    check("probabilities sum to ~1.0", abs(sum(probs) - 1.0) < 1e-6)
    check("stronger home side favoured", probs[0] > probs[2])

probs_missing = predictor.dixon_coles_probabilities("Real Madrid", "Unknown FC", "La Liga", ratings, artifacts)
check("returns None (not fabricated numbers) when a team is missing from ratings", probs_missing is None)

# ---------------------------------------------------------------
print("\nKelly cap behaviour")
kelly_cap = 0.03
bankroll = 10000.0


def kelly_stake(p, odds):
    b = odds - 1
    if b <= 0:
        return 0.0
    raw = max(0.0, (b * p - (1 - p)) / b)
    return bankroll * min(raw, kelly_cap)


huge_edge_stake = kelly_stake(0.90, 3.0)  # deliberately unrealistic edge
check("stake never exceeds the 3% kelly_cap even with a huge edge",
      huge_edge_stake <= bankroll * kelly_cap + 1e-6)

small_edge_stake = kelly_stake(0.53, 1.90)  # ~0.78% raw kelly, genuinely below the cap
check("small real edge produces a smaller stake than the cap",
      0 < small_edge_stake < bankroll * kelly_cap)

no_edge_stake = kelly_stake(0.40, 1.90)
check("no edge -> zero stake, not a fabricated positive number", no_edge_stake == 0.0)

# ---------------------------------------------------------------
print("\nCLV tracker persistence round-trip")
predictor.BASE_DIR = "/tmp"
test_path = os.path.join(predictor.BASE_DIR, "clv_tracker_2026_27.json")
if os.path.exists(test_path):
    os.remove(test_path)

entries = predictor.load_tracker()
check("starts empty when no file exists", entries == [])

entries.append({"id": 1, "match": "Test A vs Test B", "result": "WON", "profit": 50.0})
predictor.save_tracker(entries)
reloaded = predictor.load_tracker()
check("persists and reloads exactly what was saved", reloaded == entries)
os.remove(test_path)

# ---------------------------------------------------------------
print("\nlog_bet_with_clv() -- merged from the old AutomatedCLVTracker")
entries = []
e1 = predictor.log_bet_with_clv(entries, "la_liga", "Real Madrid vs Real Sociedad",
                                 "Real Madrid -1 AH", 1.90, 1.80, 250.0)
check("computes CLV when closing odds are known", e1["clv_pct"] == round(((1.90 / 1.80) - 1.0) * 100.0, 2))
check("result starts PENDING", e1["result"] == "PENDING")
check("entry is actually appended to the shared list", entries == [e1])

e2 = predictor.log_bet_with_clv(entries, "epl", "Arsenal vs Coventry",
                                 "Arsenal -1.5 AH", 1.75, None, 200.0)
check("clv_pct is null (not fabricated) when closing odds aren't known yet", e2["clv_pct"] is None)
check("ids increment off the max, not len(), avoiding collisions after deletions",
      e2["id"] == e1["id"] + 1)

# simulate a gap (as if an earlier entry were removed) and confirm no id collision
entries_with_gap = [dict(e1, id=5)]
e3 = predictor.log_bet_with_clv(entries_with_gap, "epl", "X vs Y", "Home", 2.0, None, 100.0)
check("next id is max+1 even with non-contiguous existing ids", e3["id"] == 6)

if os.path.exists(test_path):
    os.remove(test_path)

# ---------------------------------------------------------------
print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
