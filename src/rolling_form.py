from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


EMPTY_COLS = [
    "player_id",
    "rolling_sg_total",
    "rolling_sg_t2g",
    "rolling_rounds_est",
    "rolling_events_used",
    "rolling_weeks_used",
]


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=EMPTY_COLS)


def _safe_num(x, default=np.nan):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def _load_week_stats(stats_path: Path) -> pd.DataFrame:
    try:
        raw = json.loads(stats_path.read_text(encoding="utf-8"))
    except Exception:
        return pd.DataFrame()

    players = raw.get("players") or []
    rows = []
    for pl in players:
        s = pl.get("statistics") or {}
        rows.append(
            {
                "player_id": str(pl.get("id") or "").strip(),
                "events_played": _safe_num(s.get("events_played")),
                "cuts_made": _safe_num(s.get("cuts_made"), 0.0),
                "withdrawals": _safe_num(s.get("withdrawals"), 0.0),
                "strokes_gained_total": _safe_num(s.get("strokes_gained_total")),
                "strokes_gained_tee_green": _safe_num(s.get("strokes_gained_tee_green")),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df[df["player_id"] != ""].copy()
    return df


def _estimate_rounds_added(cur: pd.Series, prev: pd.Series | None) -> float:
    """
    Estimate rounds represented by a weekly snapshot increment.

    We only have cumulative season snapshots, not round logs. So we infer the
    number of new rounds between snapshots using changes in events/cuts/WD.
    """
    cur_events = _safe_num(cur.get("events_played"), 0.0)
    if prev is None:
        # First observed snapshot: infer total rounds represented so far.
        cuts = max(_safe_num(cur.get("cuts_made"), 0.0), 0.0)
        wd = max(_safe_num(cur.get("withdrawals"), 0.0), 0.0)
        missed = max(cur_events - cuts - wd, 0.0)
        rounds = (4.0 * cuts) + (2.0 * missed) + (1.0 * wd)
        return float(max(rounds, 2.0 if cur_events > 0 else 0.0))

    prev_events = _safe_num(prev.get("events_played"), 0.0)
    d_events = max(cur_events - prev_events, 0.0)
    if d_events <= 0:
        return 0.0

    d_cuts = max(_safe_num(cur.get("cuts_made"), 0.0) - _safe_num(prev.get("cuts_made"), 0.0), 0.0)
    d_wd = max(_safe_num(cur.get("withdrawals"), 0.0) - _safe_num(prev.get("withdrawals"), 0.0), 0.0)
    d_missed = max(d_events - d_cuts - d_wd, 0.0)

    rounds = (4.0 * d_cuts) + (2.0 * d_missed) + (1.0 * d_wd)

    # Fallback if the component math does not reconcile cleanly.
    if rounds <= 0:
        rounds = 4.0 * d_events

    return float(max(rounds, 1.0))


def _estimate_incremental_event_sg(cur_val: float, cur_events: float, prev_val: float | None, prev_events: float | None) -> float:
    """
    Convert cumulative SG snapshot averages into an estimated SG value for the
    newly added event block between snapshots.
    """
    if not np.isfinite(cur_val):
        return np.nan

    if prev_val is None or prev_events is None or not np.isfinite(prev_val) or not np.isfinite(prev_events):
        return float(cur_val)

    d_events = cur_events - prev_events
    if not np.isfinite(d_events) or d_events <= 0:
        return np.nan

    # Assume the stat is a cumulative average over events played.
    cur_total = cur_val * cur_events
    prev_total = prev_val * prev_events
    inc = (cur_total - prev_total) / d_events
    return float(inc)


def compute_rolling_sg_total_from_weekly(
    weekly_root: Path,
    selected_week_label: str,
    n_weeks: int = 4,
) -> pd.DataFrame:
    """
    Build a recency-weighted rolling SG signal from all available weekly folders
    up to and including the selected week.

    Because the weekly files are cumulative season snapshots rather than true
    round-by-round logs, this function reconstructs an event-block approximation
    from snapshot deltas, then weights more recent blocks more heavily.

    Returns:
      player_id, rolling_sg_total, rolling_sg_t2g, rolling_rounds_est,
      rolling_events_used, rolling_weeks_used
    """

    weekly_root = Path(weekly_root)
    if not weekly_root.exists():
        return _empty_df()

    folders = sorted([p.name for p in weekly_root.iterdir() if p.is_dir()])
    if selected_week_label not in folders:
        return _empty_df()

    selected_idx = folders.index(selected_week_label)
    use_labels = folders[: selected_idx + 1]
    if not use_labels:
        return _empty_df()

    week_frames: list[tuple[int, str, pd.DataFrame]] = []
    for order_idx, wl in enumerate(use_labels):
        p = weekly_root / wl / "player_statistics.json"
        if not p.exists():
            continue
        wk_df = _load_week_stats(p)
        if wk_df.empty:
            continue
        wk_df["week_label"] = wl
        wk_df["week_order"] = order_idx
        week_frames.append((order_idx, wl, wk_df))

    if not week_frames:
        return _empty_df()

    all_snapshots = pd.concat([wf[2] for wf in week_frames], ignore_index=True)
    if all_snapshots.empty:
        return _empty_df()

    records = []

    for player_id, grp in all_snapshots.groupby("player_id", sort=False):
        grp = grp.sort_values("week_order").reset_index(drop=True)
        prev = None
        for _, row in grp.iterrows():
            rounds_added = _estimate_rounds_added(row, prev)
            cur_events = _safe_num(row.get("events_played"), 0.0)
            prev_events = None if prev is None else _safe_num(prev.get("events_played"), 0.0)

            sg_total_inc = _estimate_incremental_event_sg(
                _safe_num(row.get("strokes_gained_total")),
                cur_events,
                None if prev is None else _safe_num(prev.get("strokes_gained_total")),
                prev_events,
            )
            sg_t2g_inc = _estimate_incremental_event_sg(
                _safe_num(row.get("strokes_gained_tee_green")),
                cur_events,
                None if prev is None else _safe_num(prev.get("strokes_gained_tee_green")),
                prev_events,
            )

            if np.isfinite(sg_total_inc) or np.isfinite(sg_t2g_inc):
                records.append(
                    {
                        "player_id": player_id,
                        "week_label": row["week_label"],
                        "week_order": int(row["week_order"]),
                        "rounds_est": float(rounds_added),
                        "sg_total_event": sg_total_inc,
                        "sg_t2g_event": sg_t2g_inc,
                    }
                )

            prev = row

    if not records:
        return _empty_df()

    inc_df = pd.DataFrame(records)
    max_order = int(inc_df["week_order"].max()) if not inc_df.empty else 0

    # Recency weights: oldest receives 1.0; newest receives the largest weight.
    # Exponential slope is intentionally moderate so we do not overfit one week.
    inc_df["recency_weight"] = np.exp(0.40 * (inc_df["week_order"] - max_order))
    inc_df["base_weight"] = inc_df["recency_weight"] * inc_df["rounds_est"].clip(lower=1.0)

    # Keep compatibility with the existing n_weeks argument by optionally
    # boosting the most recent observed blocks even more when a smaller window is
    # requested, while still using all available history.
    if n_weeks is not None:
        try:
            n_weeks_int = max(int(n_weeks), 1)
        except Exception:
            n_weeks_int = 4
        cutoff = max_order - (n_weeks_int - 1)
        recent_mask = inc_df["week_order"] >= cutoff
        inc_df.loc[recent_mask, "base_weight"] *= 1.15

    def _weighted_avg(sub: pd.DataFrame, value_col: str) -> float:
        vals = pd.to_numeric(sub[value_col], errors="coerce")
        wts = pd.to_numeric(sub["base_weight"], errors="coerce")
        mask = vals.notna() & wts.notna() & (wts > 0)
        if not mask.any():
            return np.nan
        return float(np.average(vals[mask], weights=wts[mask]))

    out = (
        inc_df.groupby("player_id", as_index=False)
        .apply(
            lambda sub: pd.Series(
                {
                    "rolling_sg_total": _weighted_avg(sub, "sg_total_event"),
                    "rolling_sg_t2g": _weighted_avg(sub, "sg_t2g_event"),
                    "rolling_rounds_est": float(pd.to_numeric(sub["rounds_est"], errors="coerce").fillna(0.0).sum()),
                    "rolling_events_used": int(sub["week_label"].nunique()),
                    "rolling_weeks_used": int(sub.shape[0]),
                }
            )
        )
        .reset_index(drop=True)
    )

    out = out[EMPTY_COLS].copy()
    out["player_id"] = out["player_id"].astype(str)
    out["rolling_sg_total"] = pd.to_numeric(out["rolling_sg_total"], errors="coerce")
    out["rolling_sg_t2g"] = pd.to_numeric(out["rolling_sg_t2g"], errors="coerce")
    out["rolling_rounds_est"] = pd.to_numeric(out["rolling_rounds_est"], errors="coerce").fillna(0.0)
    out["rolling_events_used"] = pd.to_numeric(out["rolling_events_used"], errors="coerce").fillna(0).astype(int)
    out["rolling_weeks_used"] = pd.to_numeric(out["rolling_weeks_used"], errors="coerce").fillna(0).astype(int)
    return out
