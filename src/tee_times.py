from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


def _read_json_path(p: Path) -> dict:
    return json.loads(p.read_text(encoding='utf-8'))


def _read_json_bytes(b: bytes) -> dict:
    return json.loads(b.decode('utf-8'))


def load_tee_times(path: Optional[Path] = None, file_bytes: Optional[bytes] = None) -> dict:
    if path is not None:
        return _read_json_path(path)
    if file_bytes is not None:
        return _read_json_bytes(file_bytes)
    raise ValueError('Either path or file_bytes is required.')


def tee_times_to_dataframe(raw: dict) -> pd.DataFrame:
    rows = []
    course_tz = raw.get('course_timezone')
    round_obj = raw.get('round') or {}
    for course in round_obj.get('courses', []) or []:
        for pairing in course.get('pairings', []) or []:
            tee_time_raw = pairing.get('tee_time')
            tee_dt_utc = None
            tee_dt_local = None
            try:
                tee_dt_utc = datetime.fromisoformat(str(tee_time_raw)) if tee_time_raw else None
                if tee_dt_utc is not None:
                    if course_tz:
                        tee_dt_local = pd.Timestamp(tee_dt_utc).tz_convert(course_tz)
                    else:
                        tee_dt_local = pd.Timestamp(tee_dt_utc)
            except Exception:
                tee_dt_utc = None
                tee_dt_local = None

            local_hour = tee_dt_local.hour if tee_dt_local is not None else None
            wave = None
            if local_hour is not None:
                wave = 'AM' if local_hour < 12 else 'PM'

            for player in pairing.get('players', []) or []:
                rows.append(
                    {
                        'player_id': player.get('id'),
                        'tee_time_utc': tee_dt_utc.isoformat() if tee_dt_utc else None,
                        'tee_time_local': tee_dt_local.strftime('%Y-%m-%d %I:%M %p %Z') if tee_dt_local is not None else None,
                        'tee_time_local_clock': tee_dt_local.strftime('%I:%M %p').lstrip('0') if tee_dt_local is not None else None,
                        'wave': wave,
                        'starting_hole': player.get('starting_hole'),
                        'course_name_rd1': course.get('name'),
                    }
                )

    df = pd.DataFrame(rows)
    if not df.empty and 'player_id' in df.columns:
        df = df.dropna(subset=['player_id']).drop_duplicates(subset=['player_id'], keep='first').copy()
        df['player_id'] = df['player_id'].astype(str)
    return df


def apply_wave_adjustments(
    df: pd.DataFrame,
    r1_wave_gap: float = 0.0,
    r2_wave_gap: float = 0.0,
) -> pd.DataFrame:
    """
    Positive gap means AM has the easier draw by that many strokes total.
    To keep the field mean unchanged, AM gets -gap/2 and PM gets +gap/2.

    Since round 2 typically flips AM/PM tee times, we reverse round-1 waves for R2.
    """
    out = df.copy()
    if 'wave' not in out.columns:
        out['wave'] = None

    out['wave_r1_adjust'] = 0.0
    out['wave_r2_adjust'] = 0.0

    am_mask = out['wave'].astype(str).str.upper().eq('AM')
    pm_mask = out['wave'].astype(str).str.upper().eq('PM')

    half_r1 = float(r1_wave_gap) / 2.0
    half_r2 = float(r2_wave_gap) / 2.0

    out.loc[am_mask, 'wave_r1_adjust'] = -half_r1
    out.loc[pm_mask, 'wave_r1_adjust'] = +half_r1

    # Friday usually flips the starting wave versus round 1.
    out.loc[am_mask, 'wave_r2_adjust'] = +half_r2
    out.loc[pm_mask, 'wave_r2_adjust'] = -half_r2

    out['wave_draw_summary'] = out['wave'].map({
        'AM': 'AM Thu / PM Fri',
        'PM': 'PM Thu / AM Fri',
    }).fillna('Unknown')

    return out
