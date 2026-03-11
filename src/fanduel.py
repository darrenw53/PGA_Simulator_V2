from __future__ import annotations

import itertools
from bisect import bisect_right
from dataclasses import dataclass
from typing import Optional, Tuple, Set, List

import pandas as pd


@dataclass
class LineupMeta:
    total_salary: int
    total_points: float
    n_candidates: int
    remaining_slots: int
    blend_alpha: float
    ceiling_weight: float
    leverage_weight: float
    value_salary_exp: float


def _prep_pool(sim_results: pd.DataFrame) -> pd.DataFrame:
    df = sim_results.copy()

    required = {"name", "Salary", "proj_fd_points", "FPPG"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"sim_results missing required columns: {sorted(missing)}")

    for col in [
        "Salary",
        "proj_fd_points",
        "FPPG",
        "p90_fd_points",
        "fd_ceiling_points",
        "ownership_pct",
        "leverage_score",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Salary", "proj_fd_points", "FPPG"]).copy()
    df = df[df["Salary"] > 0].copy()
    df["Salary"] = df["Salary"].astype(int)

    if "p90_fd_points" not in df.columns:
        df["p90_fd_points"] = df["proj_fd_points"]
    else:
        df["p90_fd_points"] = df["p90_fd_points"].fillna(df["proj_fd_points"])

    if "fd_ceiling_points" not in df.columns:
        df["fd_ceiling_points"] = 0.65 * df["proj_fd_points"] + 0.35 * df["p90_fd_points"]
    else:
        df["fd_ceiling_points"] = df["fd_ceiling_points"].fillna(0.65 * df["proj_fd_points"] + 0.35 * df["p90_fd_points"])

    if "ownership_pct" not in df.columns:
        df["ownership_pct"] = 10.0
    else:
        df["ownership_pct"] = df["ownership_pct"].fillna(10.0)

    if "leverage_score" not in df.columns:
        df["leverage_score"] = 0.0
    else:
        df["leverage_score"] = df["leverage_score"].fillna(0.0)

    # Ensure uniqueness
    key = "player_id" if "player_id" in df.columns else "name"
    df = df.drop_duplicates(subset=[key]).copy()

    return df.reset_index(drop=True)


def _best_under_cap_mim(
    pool: pd.DataFrame,
    cap: int,
    k: int,
    points_col: str,
) -> Optional[Tuple[List[int], int, float]]:
    if k == 0:
        return ([], 0, 0.0)

    n = len(pool)
    if n < k:
        return None

    salaries = pool["Salary"].to_numpy(dtype=int)
    points = pool[points_col].to_numpy(dtype=float)

    a = k // 2
    b = k - a

    idxs = list(range(n))

    combos_a = []
    for comb in itertools.combinations(idxs, a):
        s = int(salaries[list(comb)].sum())
        if s <= cap:
            p = float(points[list(comb)].sum())
            combos_a.append((s, p, comb))

    combos_b = []
    for comb in itertools.combinations(idxs, b):
        s = int(salaries[list(comb)].sum())
        if s <= cap:
            p = float(points[list(comb)].sum())
            combos_b.append((s, p, comb))

    if not combos_a or not combos_b:
        return None

    combos_b.sort(key=lambda x: x[0])
    b_salaries = [c[0] for c in combos_b]
    b_best_combo = []

    best_p = -1e18
    best_c = None
    for s, p, comb in combos_b:
        if p > best_p:
            best_p = p
            best_c = (s, p, comb)
        b_best_combo.append(best_c)

    best_total_p = -1e18
    best_total_s = None
    best_total_idxs = None

    for s_a, p_a, comb_a in combos_a:
        rem = cap - s_a
        j = bisect_right(b_salaries, rem) - 1
        if j < 0:
            continue

        set_a = set(comb_a)

        jj = j
        while jj >= 0:
            cand = b_best_combo[jj]
            if cand is None:
                jj -= 1
                continue
            s_b, p_b, comb_b = cand
            if set_a.isdisjoint(comb_b):
                total_s = s_a + s_b
                total_p = p_a + p_b
                if total_p > best_total_p:
                    best_total_p = total_p
                    best_total_s = total_s
                    best_total_idxs = list(comb_a) + list(comb_b)
                break
            jj -= 1

    if best_total_idxs is None:
        return None

    return (best_total_idxs, int(best_total_s), float(best_total_p))


def optimize_fanduel_lineup(
    sim_results: pd.DataFrame,
    salary_cap: int = 60000,
    lineup_size: int = 6,
    candidate_pool: int = 40,
    lock_names: Optional[Set[str]] = None,
    exclude_names: Optional[Set[str]] = None,
    blend_alpha: float = 1.0,
    ceiling_weight: float = 0.30,
    leverage_weight: float = 0.10,
    value_salary_exp: float = 0.92,
):
    """
    blend_alpha:
      0.0 => optimize purely by FPPG
      1.0 => optimize purely by proj_fd_points
      between => blended

    ceiling_weight:
      extra weight given to simulated ceiling outcomes

    leverage_weight:
      extra weight given to lower-owned / higher-leverage golfers

    value_salary_exp:
      controls how aggressively cheap salaries are rewarded in candidate selection
      1.00 = traditional points / salary
      <1.00 softens cheap-player bias
    """
    lock_names = lock_names or set()
    exclude_names = exclude_names or set()

    df = _prep_pool(sim_results)

    df = df[~df["name"].isin(exclude_names)].copy()
    if df.empty:
        return None, None

    alpha = float(blend_alpha)
    c_wt = float(ceiling_weight)
    l_wt = float(leverage_weight)
    v_exp = float(value_salary_exp)

    base_points = alpha * df["proj_fd_points"] + (1.0 - alpha) * df["FPPG"]
    df["blend_points"] = base_points
    df["optimizer_score"] = (
        base_points
        + c_wt * (df["fd_ceiling_points"] - df["proj_fd_points"])
        + l_wt * df["leverage_score"]
    )

    locked = df[df["name"].isin(lock_names)].copy()
    if len(locked) > lineup_size:
        return None, None

    locked_salary = int(locked["Salary"].sum()) if not locked.empty else 0
    locked_points = float(locked["optimizer_score"].sum()) if not locked.empty else 0.0

    if locked_salary > salary_cap:
        return None, None

    remaining_slots = lineup_size - len(locked)
    remaining_cap = salary_cap - locked_salary

    salary_denom = (df["Salary"].clip(lower=1) / 1000.0) ** max(v_exp, 0.01)
    df["value"] = df["optimizer_score"] / salary_denom

    top_points = df.sort_values("optimizer_score", ascending=False).head(candidate_pool)
    top_value = df.sort_values("value", ascending=False).head(candidate_pool)
    top_win = df.sort_values("win_pct", ascending=False).head(max(12, candidate_pool // 3)) if "win_pct" in df.columns else df.head(0)
    top_leverage = df.sort_values("leverage_score", ascending=False).head(max(12, candidate_pool // 3))
    candidates = pd.concat([top_points, top_value, top_win, top_leverage, locked], ignore_index=True)

    key = "player_id" if "player_id" in candidates.columns else "name"
    candidates = candidates.drop_duplicates(subset=[key]).copy()

    choose_pool = candidates[~candidates["name"].isin(lock_names)].copy()
    choose_pool = choose_pool.sort_values("optimizer_score", ascending=False).reset_index(drop=True)

    if remaining_slots == 0:
        lineup = locked.copy().sort_values("optimizer_score", ascending=False).reset_index(drop=True)
        meta = LineupMeta(
            total_salary=locked_salary,
            total_points=locked_points,
            n_candidates=len(candidates),
            remaining_slots=0,
            blend_alpha=alpha,
            ceiling_weight=c_wt,
            leverage_weight=l_wt,
            value_salary_exp=v_exp,
        )
        return lineup, meta.__dict__

    cheapest = choose_pool["Salary"].nsmallest(remaining_slots).sum()
    if int(cheapest) > remaining_cap:
        return None, None

    sol = _best_under_cap_mim(choose_pool, remaining_cap, remaining_slots, points_col="optimizer_score")
    if sol is None:
        return None, None

    idxs, _, _ = sol
    picked = choose_pool.iloc[idxs].copy()

    lineup = pd.concat([locked, picked], ignore_index=True)
    lineup = lineup.sort_values("optimizer_score", ascending=False).reset_index(drop=True)

    meta = LineupMeta(
        total_salary=int(lineup["Salary"].sum()),
        total_points=float(lineup["optimizer_score"].sum()),
        n_candidates=len(candidates),
        remaining_slots=remaining_slots,
        blend_alpha=alpha,
        ceiling_weight=c_wt,
        leverage_weight=l_wt,
        value_salary_exp=v_exp,
    )
    return lineup, meta.__dict__
