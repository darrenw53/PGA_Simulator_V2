import json
from pathlib import Path
import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def load_player_master(year: int) -> pd.DataFrame:
    """
    Loads player master file:
      data/player_master/players_<year>.json

    Expected structure matches your uploaded file:
      { "season": {"year": 2026}, "players": [ { "id": "...", "name": "...", ... }, ... ] }
    """
    path = Path("data") / "player_master" / f"players_{year}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing player master file: {path.as_posix()}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    players = raw.get("players", [])
    if not isinstance(players, list) or len(players) == 0:
        raise ValueError("Player master JSON has no 'players' list (or it is empty).")

    df = pd.json_normalize(players)

    # Ensure required columns exist
    if "id" not in df.columns:
        raise ValueError("Player master JSON players entries are missing 'id'.")

    # Normalize name fields
    if "name" not in df.columns:
        # build "Last, First" if possible
        fn = df.get("first_name")
        ln = df.get("last_name")
        if fn is not None or ln is not None:
            df["name"] = (df.get("last_name", "").astype(str) + ", " + df.get("first_name", "").astype(str)).str.strip(", ")
        else:
            df["name"] = df["id"].astype(str)

    df["id"] = df["id"].astype(str)
    df["name"] = df["name"].astype(str)

    return df

