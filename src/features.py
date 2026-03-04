from __future__ import annotations

import numpy as np
import pandas as pd


def make_course_fit_weights() -> dict:
    # Sensible defaults (operator can override in UI)
    return {
        "sg_total": 1.00,
        "sg_t2g": 0.75,
        "sg_putt_proxy": 0.35,
        "birdies_per_round": 0.40,
        "gir_pct": 0.20,
        "drive_avg": 0.15,
        "drive_acc": 0.10,
        "scrambling_pct": 0.20,
    }


def build_model_table(
    fanduel: pd.DataFrame,
    stats: pd.DataFrame,
    wgr: pd.DataFrame,
    rolling_sg: pd.DataFrame | None = None,
    rolling_weight: float = 0.60,
) -> pd.DataFrame:
    """
    Joins:
      FanDuel field (names + salary + FPPG)
      SportsRadar player stats (player_id + stats)
      WGR (player_id + rank)
    Matching strategy:
      1) name join (normalized to avoid punctuation/spacing issues)
      2) if you later add a master mapping file, we can do ID matching.
    """

    fd = fanduel.copy()
    st = stats.copy()
    wg = wgr.copy()

    st["name"] = (
        st["first_name"].fillna("").astype(str).str.strip()
        + " "
        + st["last_name"].fillna("").astype(str).str.strip()
    ).str.strip()

    def _clean_name(x: object) -> str:
        """Normalize player names so minor formatting differences don't break merges."""
        s = "" if x is None else str(x)
        s = s.lower().strip()
        # Common cleanup
        for ch in [",", ".", "'", "\"", "`"]:
            s = s.replace(ch, "")
        # Collapse repeated whitespace
        s = " ".join(s.split())
        return s

    fd["fd_name_clean"] = fd["fd_name"].apply(_clean_name)
    st["name_clean"] = st["name"].apply(_clean_name)

    # Name join (cleaned)
    merged = fd.merge(
        st,
        left_on="fd_name_clean",
        right_on="name_clean",
        how="left",
        suffixes=("", "_stats"),
    )

    # Some FanDuel entries might not match; keep only matched rows
    merged = merged.dropna(subset=["player_id"]).copy()

    # Drop helper columns to avoid clutter/duplication.
    for c in ["fd_name_clean", "name_clean"]:
        if c in merged.columns:
            merged = merged.drop(columns=[c])

    merged = merged.merge(wg[["player_id", "wgr_rank"]], on="player_id", how="left")

    # Optional: rolling form proxy (mean SG Total over last N weeks)
    if rolling_sg is not None and not rolling_sg.empty:
        rs = rolling_sg.copy()
        if "player_id" in rs.columns:
            rs["player_id"] = rs["player_id"].astype(str)
        if "rolling_sg_total" in rs.columns:
            rs["rolling_sg_total"] = pd.to_numeric(rs["rolling_sg_total"], errors="coerce")
        merged["player_id"] = merged["player_id"].astype(str)
        merged = merged.merge(rs[["player_id", "rolling_sg_total", "rolling_weeks_used"]], on="player_id", how="left")
    else:
        merged["rolling_sg_total"] = np.nan
        merged["rolling_weeks_used"] = 0

    # Effective SG Total used by the simulator strength model.
    # If rolling is available, blend it with the current-week season SG total.
    # If not, it falls back to the original strokes_gained_total.
    rw = float(rolling_weight)
    rw = 0.0 if not np.isfinite(rw) else max(0.0, min(1.0, rw))
    merged["strokes_gained_total_eff"] = merged.get("strokes_gained_total")
    has_roll = merged["rolling_sg_total"].notna()
    merged.loc[has_roll, "strokes_gained_total_eff"] = (
        rw * merged.loc[has_roll, "rolling_sg_total"]
        + (1.0 - rw) * merged.loc[has_roll, "strokes_gained_total"]
    )


    # Clean numeric columns
    for c in [
        "Salary", "FPPG", "wgr_rank",
        "scoring_avg", "birdies_per_round", "gir_pct", "scrambling_pct",
        "drive_avg", "drive_acc",
        "strokes_gained", "strokes_gained_total", "strokes_gained_tee_green"
    ]:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")

    # Fallbacks
    merged["wgr_rank"] = merged["wgr_rank"].fillna(999.0)
    merged["FPPG"] = merged["FPPG"].fillna(0.0)

    # If scoring_avg missing, approximate from field average
    if merged["scoring_avg"].notna().any():
        avg = float(merged["scoring_avg"].dropna().mean())
        merged["scoring_avg"] = merged["scoring_avg"].fillna(avg)
    else:
        merged["scoring_avg"] = 71.5

    return merged


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if not np.isfinite(sd) or sd == 0:
        return s * 0
    return (s - mu) / sd
