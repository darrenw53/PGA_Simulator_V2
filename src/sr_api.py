import requests
import streamlit as st

BASE = "https://api.sportradar.com"

def _get_key() -> str:
    key = st.secrets.get("SPORTSRADAR_API_KEY", "")
    if not key:
        raise RuntimeError("Missing SPORTSRADAR_API_KEY in Streamlit secrets.")
    return key

def _get(url: str) -> dict:
    key = _get_key()
    sep = "&" if "?" in url else "?"
    full = f"{url}{sep}api_key={key}"
    r = requests.get(full, headers={"accept": "application/json"}, timeout=30)
    r.raise_for_status()
    return r.json()

def pga_schedule(year: int) -> dict:
    return _get(f"{BASE}/golf/trial/pga/v3/en/{year}/tournaments/schedule.json")

def pga_player_stats(year: int) -> dict:
    return _get(f"{BASE}/golf/trial/pga/v3/en/{year}/players/statistics.json")

def wgr_rankings(year: int) -> dict:
    return _get(f"{BASE}/golf/trial/v3/en/players/wgr/{year}/rankings.json")

def tournament_scores_round(year: int, tournament_id: str, round_no: str = "01") -> dict:
    return _get(f"{BASE}/golf/trial/pga/v3/en/{year}/tournaments/{tournament_id}/rounds/{round_no}/scores.json")
