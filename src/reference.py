from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ReferencePriors:
    era_label: str
    seasons: Tuple[int, int]
    n_rows: int
    n_events: int

    # Priors / diagnostics
    suggested_round_sd: float
    winner_score_median: float
    winner_score_iqr: Tuple[float, float]  # (p25, p75)
    winner_score_p10_p90: Tuple[float, float]

    # Optional: distribution sanity checks
    made_cut_share_est: float  # rough proxy based on who has all 4 rounds


def load_reference_results_tsv(path: Path) -> pd.DataFrame:
    """
    Expected columns (from your file):
      season, start, end, tournament, location, position, name,
      score, round1, round2, round3, round4, total, earnings, fedex_points
    """
    df = pd.read_csv(path, sep="\t")

    # Normalize types
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["position_num"] = pd.to_numeric(df["position"], errors="coerce")

    # score is like "-15" in your file; convert to numeric
    df["score_num"] = pd.to_numeric(df["score"], errors="coerce")

    # rounds / total
    for c in ["round1", "round2", "round3", "round4", "total"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # event key (start + tournament is usually unique)
    df["event_key"] = df["start"].astype(str) + " | " + df["tournament"].astype(str)

    return df


def compute_reference_priors(
    df: pd.DataFrame,
    seasons: Tuple[int, int] = (2015, 2025),
    min_complete_rounds: int = 4,
) -> ReferencePriors:
    """
    Compute modern-era priors from historical results.

    suggested_round_sd:
      median of within-player round stdev for players with all 4 rounds
      (acts as a decent starting point for your simulator's 'round_sd').

    Winner score diagnostics:
      derived from 'position==1' rows and 'score_num' (relative to par).
    """
    s0, s1 = seasons
    d = df[(df["season"] >= s0) & (df["season"] <= s1)].copy()

    # --- Winner score distribution (relative to par) ---
    winners = d[d["position_num"] == 1].copy()
    winners = winners.dropna(subset=["score_num"])

    if winners.empty:
        # fallback if position parsing is weird
        winners = d[d["position"].astype(str).str.strip() == "1"].dropna(subset=["score_num"])

    w_scores = winners["score_num"].astype(float).to_numpy()
    if w_scores.size == 0:
        # last-ditch fallback
        w_scores = np.array([-15.0])

    w_p25, w_p50, w_p75 = np.percentile(w_scores, [25, 50, 75])
    w_p10, w_p90 = np.percentile(w_scores, [10, 90])

    # --- Within-player round volatility proxy ---
    round_cols = [c for c in ["round1", "round2", "round3", "round4"] if c in d.columns]
    complete = d.dropna(subset=round_cols).copy()

    # rough proxy for "made cut": has all 4 rounds recorded
    made_cut_share = float(len(complete) / max(len(d), 1))

    # within-player stdev across rounds (per event/player row)
    # (This is not perfect, but it’s a strong starting prior.)
    round_matrix = complete[round_cols].to_numpy(dtype=float)
    per_row_sd = np.std(round_matrix, axis=1, ddof=1)

    suggested_round_sd = float(np.nanmedian(per_row_sd))
    if not np.isfinite(suggested_round_sd) or suggested_round_sd <= 0:
        suggested_round_sd = 2.3  # safe fallback close to your current default

    n_events = int(d["event_key"].nunique())

    return ReferencePriors(
        era_label=f"{s0}-{s1}",
        seasons=(s0, s1),
        n_rows=int(len(d)),
        n_events=n_events,
        suggested_round_sd=suggested_round_sd,
        winner_score_median=float(w_p50),
        winner_score_iqr=(float(w_p25), float(w_p75)),
        winner_score_p10_p90=(float(w_p10), float(w_p90)),
        made_cut_share_est=made_cut_share,
    )

