from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class WeeklyData:
    schedule_raw: dict
    stats_raw: dict
    wgr_raw: dict
    fanduel_players: pd.DataFrame

    schedule_tournaments: pd.DataFrame
    player_stats: pd.DataFrame
    wgr_players: pd.DataFrame

    def get_course_meta(self, tournament_id: str) -> Optional[dict]:
        try:
            t = next(x for x in self.schedule_raw.get("tournaments", []) if x.get("id") == tournament_id)
        except Exception:
            return None

        venue = (t.get("venue") or {})
        courses = venue.get("courses") or []
        if not courses:
            return None
        c0 = courses[0] or {}
        return {
            "venue": venue.get("name"),
            "course_name": c0.get("name"),
            "yardage": c0.get("yardage"),
            "par": c0.get("par"),
        }


def list_week_folders(weekly_root: Path) -> list[str]:
    if not weekly_root.exists():
        return []
    folders = [p.name for p in weekly_root.iterdir() if p.is_dir()]
    folders.sort(reverse=True)
    return folders


def list_fanduel_csvs(folder: Path) -> list[str]:
    """Return CSV filenames in a week folder; rank likely FanDuel files first."""
    if not folder.exists():
        return []
    csvs = [p.name for p in folder.glob("*.csv")]

    def score(name: str) -> int:
        n = name.lower()
        s = 0
        if "fanduel" in n:
            s += 10
        if "pga" in n:
            s += 5
        if "player" in n:
            s += 3
        if "list" in n:
            s += 2
        return s

    csvs.sort(key=score, reverse=True)
    return csvs


def _read_json_path(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _read_json_bytes(b: bytes) -> dict:
    return json.loads(b.decode("utf-8"))


def load_weekly_data(
    folder: Optional[Path],
    schedule_bytes: Optional[bytes] = None,
    stats_bytes: Optional[bytes] = None,
    wgr_bytes: Optional[bytes] = None,
    fanduel_bytes: Optional[bytes] = None,
    fanduel_filename: Optional[str] = None,
) -> WeeklyData:
    """
    Loads weekly data from either:
      A) a folder with required jsons + a FanDuel CSV (any filename),
      B) uploaded bytes.

    Folder required:
      schedule.json
      player_statistics.json
      wgr_rankings.json
      and at least one .csv (FanDuel)
    """

    if folder is not None:
        schedule_raw = _read_json_path(folder / "schedule.json")
        stats_raw = _read_json_path(folder / "player_statistics.json")
        wgr_raw = _read_json_path(folder / "wgr_rankings.json")

        # Find FanDuel CSV
        if fanduel_filename:
            fd_path = folder / fanduel_filename
            if not fd_path.exists():
                raise FileNotFoundError(f"FanDuel CSV not found: {fd_path.as_posix()}")
        else:
            csvs = list_fanduel_csvs(folder)
            if not csvs:
                raise FileNotFoundError(
                    "No CSV found in week folder. Put your FanDuel CSV in the folder (any name)."
                )
            fd_path = folder / csvs[0]

        fanduel_df = pd.read_csv(fd_path)

    else:
        assert schedule_bytes and stats_bytes and wgr_bytes and fanduel_bytes
        schedule_raw = _read_json_bytes(schedule_bytes)
        stats_raw = _read_json_bytes(stats_bytes)
        wgr_raw = _read_json_bytes(wgr_bytes)
        fanduel_df = pd.read_csv(pd.io.common.BytesIO(fanduel_bytes))

    # ---- schedule tournaments ----
    tournaments = schedule_raw.get("tournaments", []) or []
    schedule_tournaments = pd.DataFrame(
        [
            {
                "id": t.get("id"),
                "name": t.get("name"),
                "start_date": t.get("start_date"),
                "end_date": t.get("end_date"),
                "event_type": t.get("event_type"),
                "status": t.get("status"),
            }
            for t in tournaments
        ]
    ).dropna(subset=["id", "name"])

    # ---- player stats ----
    stats_players = stats_raw.get("players", []) or []
    stat_rows = []
    for p in stats_players:
        s = p.get("statistics") or {}
        stat_rows.append(
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
    player_stats = pd.DataFrame(stat_rows).dropna(subset=["player_id"])

    # ---- WGR players ----
    wgr_players_raw = wgr_raw.get("players", []) or []
    wgr_players = pd.DataFrame(
        [
            {
                "player_id": p.get("id"),
                "wgr_first_name": p.get("first_name"),
                "wgr_last_name": p.get("last_name"),
                "wgr_rank": p.get("rank"),
            }
            for p in wgr_players_raw
        ]
    ).dropna(subset=["player_id"])

    # ---- normalize FanDuel ----
    fanduel_players = _normalize_fanduel(fanduel_df)

    # coerce numeric
    player_stats = _coerce_numeric(player_stats)
    wgr_players = _coerce_numeric(wgr_players)

    return WeeklyData(
        schedule_raw=schedule_raw,
        stats_raw=stats_raw,
        wgr_raw=wgr_raw,
        fanduel_players=fanduel_players,
        schedule_tournaments=schedule_tournaments,
        player_stats=player_stats,
        wgr_players=wgr_players,
    )


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c in {"player_id", "first_name", "last_name", "abbr_name", "wgr_first_name", "wgr_last_name"}:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _normalize_fanduel(fd: pd.DataFrame) -> pd.DataFrame:
    df = fd.copy()
    df.columns = [c.strip() for c in df.columns]

    rename_map = {}
    for col in df.columns:
        lc = col.lower().strip()
        if lc == "first name":
            rename_map[col] = "First Name"
        elif lc == "last name":
            rename_map[col] = "Last Name"
        elif lc == "salary":
            rename_map[col] = "Salary"
        elif lc == "fppg":
            rename_map[col] = "FPPG"
        elif lc == "id":
            rename_map[col] = "Id"
    df = df.rename(columns=rename_map)

    needed = ["Id", "First Name", "Last Name", "Salary", "FPPG"]
    for n in needed:
        if n not in df.columns:
            raise ValueError(f"FanDuel CSV missing required column: {n}")

    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
    df["FPPG"] = pd.to_numeric(df["FPPG"], errors="coerce")
    df = df.dropna(subset=["Salary"]).copy()

    df["fd_name"] = (
        df["First Name"].fillna("").astype(str).str.strip()
        + " "
        + df["Last Name"].fillna("").astype(str).str.strip()
    ).str.strip()

    return df
