from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .features import zscore


@dataclass
class SimConfig:
    n_sims: int = 25000
    rng_seed: int | None = None
    cut_size: int = 65
    round_sd: float = 2.3
    course_difficulty: float = 0.0
    wgr_weight: float = 1.0
    course_fit_weights: dict | None = None


def _compute_strength(df: pd.DataFrame, cfg: SimConfig) -> pd.Series:
    """
    Builds a single 'strength' z-score using the stats you actually have.
    Positive = better (lower scores expected).
    """
    w = cfg.course_fit_weights or {}

    # Higher SG is better => positive strength
    sg_total = zscore(
        df.get(
            "strokes_gained_total_eff",
            df.get("strokes_gained_total", pd.Series([0] * len(df))),
        )
    )
    sg_t2g = zscore(
        df.get(
            "strokes_gained_tee_green_eff",
            df.get("strokes_gained_tee_green", pd.Series([0] * len(df))),
        )
    )
    sg_putt_proxy = zscore(
        df.get("strokes_gained", pd.Series([0] * len(df)))
    )  # proxy, since we don't have SG:Putting split
    birdies = zscore(df.get("birdies_per_round", pd.Series([0] * len(df))))
    gir = zscore(df.get("gir_pct", pd.Series([0] * len(df))))
    drive = zscore(df.get("drive_avg", pd.Series([0] * len(df))))
    acc = zscore(df.get("drive_acc", pd.Series([0] * len(df))))
    scramble = zscore(df.get("scrambling_pct", pd.Series([0] * len(df))))

    # WGR: lower rank is better, so invert
    wgr_rank = pd.to_numeric(df.get("wgr_rank", 999), errors="coerce").fillna(999.0)
    wgr_strength = zscore(-wgr_rank) * float(cfg.wgr_weight)

    strength = (
        float(w.get("sg_total", 0.0)) * sg_total
        + float(w.get("sg_t2g", 0.0)) * sg_t2g
        + float(w.get("sg_putt_proxy", 0.0)) * sg_putt_proxy
        + float(w.get("birdies_per_round", 0.0)) * birdies
        + float(w.get("gir_pct", 0.0)) * gir
        + float(w.get("drive_avg", 0.0)) * drive
        + float(w.get("drive_acc", 0.0)) * acc
        + float(w.get("scrambling_pct", 0.0)) * scramble
        + wgr_strength
    )

    # Normalize strength so typical range isn't crazy
    strength = zscore(strength)
    return strength.fillna(0.0)


def _effective_round_sd(round_sd: float) -> float:
    """
    Compress very high volatility settings so randomness cannot completely swamp
    player skill. Values up to 2.8 are unchanged; above that they are softened.
    """
    try:
        rsd = float(round_sd)
    except Exception:
        rsd = 2.3

    if not np.isfinite(rsd):
        return 2.3
    if rsd <= 2.8:
        return rsd

    # Example:
    # 3.0 -> 2.87
    # 3.4 -> 3.01
    # 4.0 -> 3.22
    return 2.8 + 0.35 * (rsd - 2.8)


def simulate_tournament(players: pd.DataFrame, cfg: SimConfig) -> pd.DataFrame:
    df = players.copy().reset_index(drop=True)

    rng = np.random.default_rng(cfg.rng_seed)

    # baseline per-round scoring avg
    base_mu = pd.to_numeric(df["scoring_avg"], errors="coerce").fillna(71.5).to_numpy()

    strength = _compute_strength(df, cfg).to_numpy()

    # translate strength -> strokes improvement
    # Increased from 0.6 to 0.72 to improve elite-player separation and win calibration.
    # 1.0 strength ≈ 0.72 strokes/round better.
    mu = base_mu - 0.72 * strength + float(cfg.course_difficulty)

    n = len(df)
    sims = int(cfg.n_sims)

    # Optional round-specific wave/weather adjustments (strokes).
    # Positive values make scoring worse; negative values help scoring.
    round_adjustments = np.zeros((n, 4), dtype=float)
    for idx, col in enumerate(["wave_r1_adjust", "wave_r2_adjust", "wave_r3_adjust", "wave_r4_adjust"]):
        if col in df.columns:
            round_adjustments[:, idx] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()

    round_loc = mu[:, None] + round_adjustments

    # Apply elite-preserving variance shaping without changing the UI control.
    eff_round_sd = _effective_round_sd(cfg.round_sd)

    # Stronger players get a slightly tighter scoring distribution.
    # Caps at 12% reduction so this helps elites without overfitting.
    strength_clip = np.clip(strength, -3.0, 3.0)
    player_sd_scale = 1.0 - np.clip(strength_clip, 0.0, None) * 0.04
    player_sd_scale = np.clip(player_sd_scale, 0.88, 1.00)
    player_round_sd = eff_round_sd * player_sd_scale

    # Rounds: [sims, n, 4]
    rounds = rng.normal(
        loc=round_loc[None, :, :],
        scale=player_round_sd[None, :, None],
        size=(sims, n, 4),
    )

    # Total score (4 rounds)
    totals = rounds.sum(axis=2)

    # Cut after 2 rounds
    r2_totals = rounds[:, :, :2].sum(axis=2)

    # Determine who makes cut per sim
    # lower is better
    cut_mask = np.zeros((sims, n), dtype=bool)
    for i in range(sims):
        order = np.argsort(r2_totals[i, :])
        # ties: keep simple by taking top N exactly
        cut_mask[i, order[: int(cfg.cut_size)]] = True

    # Make missed-cut players comparable to 4-round totals
    totals_adj = totals.copy()

    # Penalty round score: field-average expected round + 2 strokes
    penalty_round = float(np.nanmean(mu)) + 2.0
    miss_cut_penalty = 2.0 * penalty_round  # rounds 3 and 4

    for i in range(sims):
        miss = ~cut_mask[i, :]
        totals_adj[i, miss] = r2_totals[i, miss] + miss_cut_penalty

    # Finish rank (1 = best)
    finish_rank = np.argsort(np.argsort(totals_adj, axis=1), axis=1) + 1

    # Metrics
    win = (finish_rank == 1).mean(axis=0)
    top10 = (finish_rank <= 10).mean(axis=0)
    make_cut = cut_mask.mean(axis=0)
    avg_finish = finish_rank.mean(axis=0)
    avg_total_score = totals_adj.mean(axis=0)

    # Project FanDuel points (approx)
    fppg = pd.to_numeric(df.get("FPPG", 0.0), errors="coerce").fillna(0.0).to_numpy()
    proj_fd = (
        0.70 * fppg
        + 35.0 * win
        + 12.0 * top10
        + 8.0 * make_cut
        - 0.08 * (avg_finish - 1.0)
    )

    out = df.copy()
    out["win_pct"] = win * 100.0
    out["top10_pct"] = top10 * 100.0
    out["make_cut_pct"] = make_cut * 100.0
    out["avg_finish"] = avg_finish
    out["avg_total_score"] = avg_total_score
    out["proj_fd_points"] = proj_fd

    # Keep clean player name
    if "name" not in out.columns:
        out["name"] = (
            out["First Name"].fillna("").astype(str)
            + " "
            + out["Last Name"].fillna("").astype(str)
        ).str.strip()

    # Sort by win% by default
    out = out.sort_values(["win_pct", "proj_fd_points"], ascending=[False, False]).reset_index(drop=True)
    return out
