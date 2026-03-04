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
    sg_total = zscore(df.get("strokes_gained_total_eff", df.get("strokes_gained_total", pd.Series([0] * len(df)))))
    sg_t2g = zscore(df.get("strokes_gained_tee_green", pd.Series([0] * len(df))))
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


def simulate_tournament(players: pd.DataFrame, cfg: SimConfig) -> pd.DataFrame:
    df = players.copy().reset_index(drop=True)

    rng = np.random.default_rng(cfg.rng_seed)

    # baseline per-round scoring avg
    base_mu = pd.to_numeric(df["scoring_avg"], errors="coerce").fillna(71.5).to_numpy()

    strength = _compute_strength(df, cfg).to_numpy()

    # translate strength -> strokes improvement
    # 1.0 strength ≈ 0.6 strokes/round better (tunable later)
    mu = base_mu - 0.6 * strength + float(cfg.course_difficulty)

    n = len(df)
    sims = int(cfg.n_sims)

    # Rounds: [sims, n, 4]
    rounds = rng.normal(
        loc=mu[None, :, None], scale=float(cfg.round_sd), size=(sims, n, 4)
    )
    # Total score (4 rounds)
    totals = rounds.sum(axis=2)

    # Cut after 2 rounds:
    r2_totals = rounds[:, :, :2].sum(axis=2)

    # Determine who makes cut per sim
    # lower is better
    cut_mask = np.zeros((sims, n), dtype=bool)
    for i in range(sims):
        order = np.argsort(r2_totals[i, :])
        # ties: keep simple by taking top N exactly
        cut_mask[i, order[: int(cfg.cut_size)]] = True

    # ---------------------------------------------------------------------
    # FIX: Make missed-cut players comparable to 4-round totals
    #
    # Previously, missed-cut players were assigned only their 2-round total,
    # which is ~140 and therefore "beats" any 4-round total (~260-290).
    # That caused missed-cut players to win sims and could bury elite players.
    #
    # Correct approach: assign a 4-round-equivalent total by adding a penalty
    # for rounds 3-4 for anyone who misses the cut.
    # ---------------------------------------------------------------------
    totals_adj = totals.copy()

    # Penalty round score: field-average expected round + 2 strokes (conservative)
    penalty_round = float(np.nanmean(mu)) + 2.0
    miss_cut_penalty = 2.0 * penalty_round  # rounds 3 and 4

    for i in range(sims):
        miss = ~cut_mask[i, :]
        totals_adj[i, miss] = r2_totals[i, miss] + miss_cut_penalty

    # Finish rank (1 = best). Now totals_adj is comparable for all players.
    finish_rank = np.argsort(np.argsort(totals_adj, axis=1), axis=1) + 1

    # Metrics
    win = (finish_rank == 1).mean(axis=0)
    top10 = (finish_rank <= 10).mean(axis=0)
    make_cut = cut_mask.mean(axis=0)
    avg_finish = finish_rank.mean(axis=0)
    avg_total_score = totals_adj.mean(axis=0)

    # Project FanDuel points (approx):
    # - Use FanDuel FPPG as base, then adjust for upside + cut risk using sim outputs.
    fppg = pd.to_numeric(df.get("FPPG", 0.0), errors="coerce").fillna(0.0).to_numpy()
    # Upside boosts
    proj_fd = (
        0.70 * fppg
        + 35.0 * win  # winning is huge
        + 12.0 * top10  # top10 meaningful
        + 8.0 * make_cut  # survive cut matters
        - 0.08 * (avg_finish - 1.0)  # small penalty for worse average finish
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
