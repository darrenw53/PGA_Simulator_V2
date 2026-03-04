
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import json

from .file_loader import list_week_folders


# Stats where LOWER is better in golf (we invert them to "higher is better")
LOWER_IS_BETTER = {
    "scoring_avg",
    "putt_avg",
    "world_rank",
    "total_driving",  # typically a rank-like metric
    "wgr_rank",
}


@dataclass
class HotnessResult:
    """Computed hotness + per-week form scores for a set of players."""
    weeks: List[str]                 # week folder labels, oldest -> newest
    form_scores_wide: pd.DataFrame   # columns: player_id, player_name, hotness_raw, hotness_1_5, w0..wN lists


def _parse_week_label(label: str) -> Optional[pd.Timestamp]:
    try:
        return pd.to_datetime(label, errors="raise")
    except Exception:
        return None




def _read_player_stats(stats_path: Path) -> pd.DataFrame:
    """Read a SportsRadar-style player_statistics.json and return the same columns as file_loader.load_weekly_data()."""
    if not stats_path.exists():
        return pd.DataFrame(columns=["player_id", "first_name", "last_name", "abbr_name"])

    raw = json.loads(stats_path.read_text(encoding="utf-8"))
    stats_players = raw.get("players", []) or []
    rows = []
    for p in stats_players:
        s = p.get("statistics") or {}
        rows.append(
            {
                "player_id": p.get("id"),
                "first_name": p.get("first_name"),
                "last_name": p.get("last_name"),
                "abbr_name": p.get("abbr_name"),
                "scoring_avg": s.get("scoring_avg"),
                "birdies_per_round": s.get("birdies_per_round"),
                "gir_pct": s.get("gir_pct"),
                "scrambling_pct": s.get("scrambling_pct"),
                "drive_avg": s.get("drive_avg"),
                "drive_acc": s.get("drive_acc"),
                "putt_avg": s.get("putt_avg"),
                "sand_saves_pct": s.get("sand_saves_pct"),
                "total_driving": s.get("total_driving"),
                "world_rank": s.get("world_rank"),
                "strokes_gained": s.get("strokes_gained"),
                "strokes_gained_total": s.get("strokes_gained_total"),
                "strokes_gained_tee_green": s.get("strokes_gained_tee_green"),
            }
        )

    df = pd.DataFrame(rows).dropna(subset=["player_id"])
    # coerce numeric where possible
    for c in df.columns:
        if c in {"player_id", "first_name", "last_name", "abbr_name"}:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def pick_last_n_weeks(all_week_labels_desc: List[str], selected_week_label: str, n: int = 4) -> List[str]:
    """
    all_week_labels_desc: labels sorted newest -> oldest
    Returns up to n labels ending at selected_week_label, sorted oldest -> newest.
    """
    if not all_week_labels_desc:
        return []

    # keep only parsable date labels
    parsed = [(w, _parse_week_label(w)) for w in all_week_labels_desc]
    parsed = [(w, d) for (w, d) in parsed if d is not None]

    if not parsed:
        return []

    # sort newest -> oldest (input likely already), but enforce
    parsed.sort(key=lambda x: x[1], reverse=True)
    labels_desc = [w for (w, _) in parsed]

    if selected_week_label not in labels_desc:
        # fallback: just take the latest n
        chosen = labels_desc[:n]
    else:
        idx = labels_desc.index(selected_week_label)
        chosen = labels_desc[idx: idx + n]

    # Now chosen is newest->older because labels_desc is newest->oldest and we sliced forward
    # Actually if idx is in that list, slicing forward goes toward older weeks.
    # We want up to n weeks ending at the selected week (newest) going backward.
    # Example labels_desc: [Feb05, Jan29, Jan22, Jan15]
    # selected=Feb05 idx=0 -> slice 0:4 -> [Feb05, Jan29, Jan22, Jan15] good
    # selected=Jan22 idx=2 -> slice 2:6 -> [Jan22, Jan15] good (older only)
    # Then reverse to oldest->newest.
    chosen = list(reversed(chosen))
    return chosen


def _zscore(s: pd.Series) -> pd.Series:
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if sd is None or np.isnan(sd) or sd == 0:
        return s * 0.0
    return (s - mu) / sd


def compute_week_form_scores(player_stats: pd.DataFrame, stat_cols: List[str]) -> pd.Series:
    """
    Build a single 'form score' per player for a week:
      - orient each stat so higher is better
      - z-score within the week field
      - average z-scores across available stats
    Returns a Series indexed by player_id.
    """
    df = player_stats.copy()

    # ensure numeric
    for c in stat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    oriented = {}
    for c in stat_cols:
        s = df[c]
        if c in LOWER_IS_BETTER:
            s = -1.0 * s
        oriented[c] = _zscore(s)

    zdf = pd.DataFrame(oriented)
    form = zdf.mean(axis=1, skipna=True)
    form.index = df["player_id"].astype(str).values
    return form


def compute_hotness_from_forms(form_matrix: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    form_matrix: rows=player_id, cols=week labels (oldest->newest), values=form score
    Returns:
      hotness_raw (slope),
      hotness_1_5 (quintile rating, 1..5)
    """
    # slope of form score across weeks for each player
    xs = np.arange(form_matrix.shape[1], dtype=float)

    def slope(row: np.ndarray) -> float:
        mask = ~np.isnan(row)
        if mask.sum() < 2:
            return np.nan
        x = xs[mask]
        y = row[mask]
        # simple least-squares slope
        x = x - x.mean()
        denom = (x * x).sum()
        if denom == 0:
            return 0.0
        return float((x * (y - y.mean())).sum() / denom)

    hot_raw = form_matrix.apply(lambda r: slope(r.values.astype(float)), axis=1)

    # rating via quintiles among players with a value
    valid = hot_raw.dropna()
    if len(valid) == 0:
        hot_1_5 = hot_raw.apply(lambda _: np.nan)
    else:
        # pandas qcut can fail with many ties; handle robustly
        try:
            bins = pd.qcut(valid, q=5, labels=[1, 2, 3, 4, 5])
            hot_1_5 = hot_raw.copy()
            hot_1_5.loc[bins.index] = bins.astype(float)
        except Exception:
            # fallback: rank percentiles
            ranks = valid.rank(method="average", pct=True)
            hot_1_5 = hot_raw.copy()
            hot_1_5.loc[ranks.index] = (np.ceil(ranks * 5)).clip(1, 5)

    return hot_raw, hot_1_5


def compute_hotness_last_n_weeks(
    weekly_root: Path,
    selected_week_label: str,
    n_weeks: int = 4,
) -> HotnessResult:
    """
    Loads player_statistics.json from the last N week folders (ending at selected_week_label),
    computes per-week form scores and a 1-5 hotness rating per golfer.

    This is informational only and does NOT alter simulation calculations.
    """
    all_weeks_desc = list_week_folders(weekly_root)  # newest -> oldest
    weeks = pick_last_n_weeks(all_weeks_desc, selected_week_label, n=n_weeks)  # oldest -> newest
    if not weeks:
        return HotnessResult(weeks=[], form_scores_wide=pd.DataFrame())

    # Decide which stat columns to use based on loader's player_stats columns
    # (we'll intersect across weeks so missing cols don't break)
    week_dfs: Dict[str, pd.DataFrame] = {}
    stat_cols: Optional[List[str]] = None

    for w in weeks:
        ps = _read_player_stats(weekly_root / w / "player_statistics.json")
        week_dfs[w] = ps

        cols = [c for c in ps.columns if c not in {"player_id", "first_name", "last_name", "abbr_name"}]
        # keep numeric-ish stat columns only
        cols = [c for c in cols if ps[c].dtype != "object"]
        if stat_cols is None:
            stat_cols = cols
        else:
            stat_cols = [c for c in stat_cols if c in cols]

    stat_cols = stat_cols or []
    # drop obviously non-performance columns if present
    stat_cols = [c for c in stat_cols if c not in {"drive_avg"} or True]  # keep drive_avg

    # Build form matrix: player_id x week
    forms = {}
    names = {}

    for w in weeks:
        ps = week_dfs[w]
        ps["player_id"] = ps["player_id"].astype(str)
        ps["player_name"] = (ps["first_name"].fillna("").astype(str).str.strip() + " " + ps["last_name"].fillna("").astype(str).str.strip()).str.strip()
        for pid, pname in zip(ps["player_id"], ps["player_name"]):
            if pid and pname and pid not in names:
                names[pid] = pname

        form = compute_week_form_scores(ps, stat_cols)
        forms[w] = form

    form_matrix = pd.DataFrame(forms).reindex(columns=weeks)  # index=player_id
    # Add any names we saw
    name_series = pd.Series(names, name="player_name")
    form_matrix = form_matrix.join(name_series, how="left")

    # compute hotness
    form_only = form_matrix[weeks].astype(float)
    hot_raw, hot_1_5 = compute_hotness_from_forms(form_only)

    out = pd.DataFrame({
        "player_id": form_matrix.index.astype(str),
        "player_name": form_matrix["player_name"].fillna(""),
        "hotness_raw": hot_raw.values,
        "hotness_1_5": hot_1_5.values,
    })
    # store sparkline as list (Streamlit LineChartColumn expects list-like)
    out["form_last4"] = form_only.apply(lambda r: [None if np.isnan(x) else float(x) for x in r.values], axis=1).values
    out = out.sort_values(["hotness_1_5", "hotness_raw"], ascending=[False, False], na_position="last").reset_index(drop=True)

    return HotnessResult(weeks=weeks, form_scores_wide=out)
