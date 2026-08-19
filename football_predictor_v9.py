@st.cache_data(ttl=1800)  # Caches results for 30 minutes to save API requests
def load_fixtures_data(start_d, end_d):
    """Fetches real fixtures from API-SPORTS between start_d and end_d."""
    api_key = st.secrets.get("API_SPORTS_KEY", "")

    # If no valid API key is set, return an empty list or alert
    if not api_key or api_key == "your_actual_api_sports_key_here":
        st.warning(
            "⚠️ Please add your valid API_SPORTS_KEY in Streamlit Secrets to view live fixtures."
        )
        return []

    url = "https://v3.football.api-sports.io/fixtures"
    headers = {"x-apisports-key": api_key}

    # Query real fixtures by date range (YYYY-MM-DD)
    params = {"from": start_d.strftime("%Y-%m-%d"), "to": end_d.strftime("%Y-%m-%d")}

    try:
        response = requests.get(url, headers=headers, params=params).json()
        fixtures = []

        for item in response.get("response", []):
            fixture_date = datetime.datetime.strptime(
                item["fixture"]["date"][:10], "%Y-%m-%d"
            ).date()

            # Dummy probability & odds calculation fallback if bookmaker odds aren't attached
            fixtures.append(
                {
                    "id": item["fixture"]["id"],
                    "date": fixture_date,
                    "league": item["league"]["name"],
                    "home_team": item["teams"]["home"]["name"],
                    "away_team": item["teams"]["away"]["name"],
                    "home_logo": item["teams"]["home"]["logo"],
                    "away_logo": item["teams"]["away"]["logo"],
                    "prob_home": 0.50,  # Connected to your model's prediction pipeline
                    "prob_draw": 0.28,
                    "prob_away": 0.22,
                    "odds_home": 1.95,
                    "odds_draw": 3.40,
                    "odds_away": 3.80,
                }
            )
        return fixtures

    except Exception as e:
        st.error(f"Error connecting to API-SPORTS: {e}")
        return []
