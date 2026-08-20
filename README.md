# Football Predictor V9 — SportPesa Value Engine

Streamlit app for match predictions, market-derived value bets, and CLV tracking
across La Liga, EPL, Serie A, Bundesliga, Ligue 1.

## Files
- `sportpesa_predictor_v2.py` — the app (run this one)
- `football_predictor_v9_DEPRECATED.py` — old standalone engine, merged in, kept only as a pointer stub
- `v9_params.json`, `league_home_advantage.json`, `isotonic_calibration.pkl` — model artifacts
- `team_ratings_TEMPLATE.json` — schema for the rolling-xG team ratings the Dixon-Coles model needs (not yet populated — see below)
- `clv_tracker_2026_27.json` — bet log (starts with whatever's already in it; delete/empty it for a fresh season if you don't want prior entries in a public repo)
- `test_predictor_logic.py` — unit tests for the core logic, no network/Streamlit runtime required

## Local setup
```bash
git clone <your-repo-url>
cd <repo>
pip install -r requirements.txt

cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then edit .streamlit/secrets.toml and put your real API-SPORTS key in it
# (this file is gitignored — it will never be committed)

python3 test_predictor_logic.py   # optional: confirm logic still passes locally
streamlit run sportpesa_predictor_v2.py
```

## Model status
Predictions currently fall back to de-vigged market-implied probability
because `team_ratings.json` (real per-team rolling xG, not the template)
hasn't been supplied yet. Drop a populated copy matching
`team_ratings_TEMPLATE.json`'s schema next to the app to switch fixtures
over to the real Dixon-Coles model — the app already knows how to load
and use it the moment it exists.

## Deploying (Streamlit Community Cloud)
1. Push this repo to GitHub (public or private).
2. Go to https://share.streamlit.io → "New app" → pick this repo/branch → main file `sportpesa_predictor_v2.py`.
3. In the app's **Settings → Secrets**, paste:
   ```
   API_SPORTS_KEY = "your_actual_api_sports_key_here"
   ```
   (this is separate from and takes the place of your local `secrets.toml`)
4. Deploy. Community Cloud auto-redeploys on every push to the connected branch.
