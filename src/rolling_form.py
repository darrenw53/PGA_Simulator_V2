from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd


def compute_rolling_sg_total_from_weekly(
    weekly_root: Path,
    selected_week_label: str,
    n_weeks: int = 4,
) -> pd.DataFrame:
    """
    Build a simple rolling "form" proxy from the last N week folders (including the selected week).

    We read each week folder's player_statistics.json and extract:
      - player_id
      - strokes_gained_total

    Then compute a per-player mean across the available weeks.

    Returns a dataframe:
      player_id, rolling_sg_total, rolling_weeks_used
    """

    weekly_root = Path(weekly_root)
    if not weekly_root.exists():
        return pd.DataFrame(columns=["player_id", "rolling_sg_total", "rolling_weeks_used"])

    # Week folders are assumed to be named sortable (your app sorts reverse for newest-first)
    folders = [p.name for p in weekly_root.iterdir() if p.is_dir()]
    folders.sort(reverse=True)

    if selected_week_label not in folders:
        return pd.DataFrame(columns=["player_id", "rolling_sg_total", "rolling_weeks_used"])

    start_idx = folders.index(selected_week_label)
    use_labels = folders[start_idx : start_idx + int(n_weeks)]
    rows = []

    for wl in use_labels:
        p = weekly_root / wl / "player_statistics.json"
        if not p.exists():
            continue
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        for pl in (raw.get("players") or []):
            s = pl.get("statistics") or {}
            rows.append(
                {
                    "week_label": wl,
                    "player_id": str(pl.get("id")),
                    "strokes_gained_total": s.get("strokes_gained_total"),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["player_id", "rolling_sg_total", "rolling_weeks_used"])

    df = pd.DataFrame(rows)
    df["strokes_gained_total"] = pd.to_numeric(df["strokes_gained_total"], errors="coerce")
    df = df.dropna(subset=["player_id"])

    g = df.groupby("player_id", as_index=False).agg(
        rolling_sg_total=("strokes_gained_total", "mean"),
        rolling_weeks_used=("strokes_gained_total", "count"),
    )
    g["rolling_sg_total"] = pd.to_numeric(g["rolling_sg_total"], errors="coerce")
    g["rolling_weeks_used"] = pd.to_numeric(g["rolling_weeks_used"], errors="coerce").fillna(0).astype(int)

    return g
