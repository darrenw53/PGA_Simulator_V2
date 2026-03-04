from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd


def slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")[:60] or "event"


def utc_run_id() -> str:
    # Stable, sortable, unique enough
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


@dataclass
class RunRecord:
    tournament_folder: Path
    run_folder: Path
    run_id: str
    tournament_id: str
    tournament_name: str
    created_utc: str
    settings_path: Path
    predictions_path: Path
    actuals_path: Optional[Path]


def ensure_runs_root(repo_root: Path) -> Path:
    runs_root = repo_root / "data" / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    return runs_root


def make_run_paths(
    runs_root: Path,
    tournament_id: str,
    tournament_name: str,
    run_id: Optional[str] = None,
) -> tuple[Path, Path]:
    event_folder = runs_root / f"{tournament_id}_{slugify(tournament_name)}"
    event_folder.mkdir(parents=True, exist_ok=True)

    rid = run_id or utc_run_id()
    run_folder = event_folder / rid
    run_folder.mkdir(parents=True, exist_ok=True)

    return event_folder, run_folder


def save_run(
    repo_root: Path,
    tournament_id: str,
    tournament_name: str,
    settings: Dict[str, Any],
    predictions: pd.DataFrame,
    run_id: Optional[str] = None,
) -> RunRecord:
    runs_root = ensure_runs_root(repo_root)
    event_folder, run_folder = make_run_paths(runs_root, tournament_id, tournament_name, run_id=run_id)

    settings_path = run_folder / "settings.json"
    predictions_path = run_folder / "predictions.csv"
    actuals_path = run_folder / "actuals.csv"

    # Add basic metadata
    settings_out = dict(settings)
    settings_out.setdefault("tournament_id", tournament_id)
    settings_out.setdefault("tournament_name", tournament_name)
    settings_out.setdefault("run_id", run_folder.name)
    settings_out.setdefault("created_utc", datetime.utcnow().isoformat(timespec="seconds") + "Z")

    settings_path.write_text(json.dumps(settings_out, indent=2), encoding="utf-8")
    predictions.to_csv(predictions_path, index=False)

    return RunRecord(
        tournament_folder=event_folder,
        run_folder=run_folder,
        run_id=run_folder.name,
        tournament_id=str(tournament_id),
        tournament_name=str(tournament_name),
        created_utc=settings_out["created_utc"],
        settings_path=settings_path,
        predictions_path=predictions_path,
        actuals_path=actuals_path if actuals_path.exists() else None,
    )


def list_runs(repo_root: Path) -> List[RunRecord]:
    runs_root = ensure_runs_root(repo_root)
    out: List[RunRecord] = []

    if not runs_root.exists():
        return out

    for event_folder in sorted([p for p in runs_root.iterdir() if p.is_dir()], reverse=True):
        # event_folder name: "<tournament_id>_<slug>"
        tournament_id = event_folder.name.split("_", 1)[0]

        for run_folder in sorted([p for p in event_folder.iterdir() if p.is_dir()], reverse=True):
            settings_path = run_folder / "settings.json"
            predictions_path = run_folder / "predictions.csv"
            actuals_path = run_folder / "actuals.csv"

            if not settings_path.exists() or not predictions_path.exists():
                continue

            try:
                settings = json.loads(settings_path.read_text(encoding="utf-8"))
            except Exception:
                settings = {}

            tournament_name = settings.get("tournament_name", event_folder.name)
            created_utc = settings.get("created_utc", "")

            out.append(
                RunRecord(
                    tournament_folder=event_folder,
                    run_folder=run_folder,
                    run_id=run_folder.name,
                    tournament_id=str(settings.get("tournament_id", tournament_id)),
                    tournament_name=str(tournament_name),
                    created_utc=str(created_utc),
                    settings_path=settings_path,
                    predictions_path=predictions_path,
                    actuals_path=actuals_path if actuals_path.exists() else None,
                )
            )

    return out


def load_predictions(run: RunRecord) -> pd.DataFrame:
    return pd.read_csv(run.predictions_path)


def load_settings(run: RunRecord) -> Dict[str, Any]:
    return json.loads(run.settings_path.read_text(encoding="utf-8"))


def save_actuals_csv(run: RunRecord, csv_bytes: bytes) -> Path:
    path = run.run_folder / "actuals.csv"
    path.write_bytes(csv_bytes)
    return path

