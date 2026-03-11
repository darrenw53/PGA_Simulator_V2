from __future__ import annotations

from pathlib import Path
import streamlit as st
import pandas as pd

from src.file_loader import WeeklyData, load_weekly_data, list_week_folders, list_fanduel_csvs
from src.features import build_model_table, make_course_fit_weights
from src.simulator import SimConfig, simulate_tournament
from src.fanduel import optimize_fanduel_lineup
from src.run_store import list_runs, load_predictions, save_run, save_actuals_csv
from src.hotness import compute_hotness_last_n_weeks
from src.rolling_form import compute_rolling_sg_total_from_weekly
from src.tee_times import load_tee_times, tee_times_to_dataframe, apply_wave_adjustments

# NEW: early-season reference priors
from src.reference import load_reference_results_tsv, compute_reference_priors


APP_TITLE = "SignalAI • PGA Simulator"
PASSWORD = "signalai123"


def password_gate() -> bool:
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False

    if st.session_state.auth_ok:
        return True

    st.title(APP_TITLE)
    st.subheader("Login")
    pwd = st.text_input("Password", type="password")
    if st.button("Enter", use_container_width=True):
        if pwd == PASSWORD:
            st.session_state.auth_ok = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


def _safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default


def _format_money(x):
    try:
        return f"${int(x):,}"
    except Exception:
        return str(x)




@st.cache_data(show_spinner=False)
def _compute_hotness(weekly_root_str: str, selected_week_label: str):
    return compute_hotness_last_n_weeks(Path(weekly_root_str), selected_week_label)

@st.cache_data(show_spinner=False)
def _compute_rolling_sg(weekly_root_str: str, selected_week_label: str, n_weeks: int = 4):
    return compute_rolling_sg_total_from_weekly(Path(weekly_root_str), selected_week_label, n_weeks=n_weeks)
@st.cache_data(show_spinner=False)
def _load_reference_df(path_str: str):
    return load_reference_results_tsv(Path(path_str))


@st.cache_data(show_spinner=False)
def _load_tee_times_from_path(path_str: str):
    raw = load_tee_times(path=Path(path_str))
    return tee_times_to_dataframe(raw)

@st.cache_data(show_spinner=False)
def _load_tee_times_from_bytes(file_bytes: bytes):
    raw = load_tee_times(file_bytes=file_bytes)
    return tee_times_to_dataframe(raw)


def _heater_meter_col_config():
    """Column config used anywhere we display the 1–5 hotness score."""
    return st.column_config.ProgressColumn(
        "Heater Meter (1–5)",
        min_value=1,
        max_value=5,
        help="Informational only: 1 = colder (recent form declining), 5 = hotter (recent form improving).",
    )


def _attach_heater_meter(df, heater_map_df):
    """Add Heater Meter column to a dataframe if we have hotness data."""
    if df is None:
        return df
    try:
        if heater_map_df is None or heater_map_df.empty:
            out = df.copy()
            if "Heater Meter" not in out.columns:
                out["Heater Meter"] = None
            return out

        out = df.copy()
        # If this DF already has a Heater Meter column, drop it before merging/renaming.
        # Otherwise, renaming hotness_1_5 -> Heater Meter would create duplicate column names.
        if "Heater Meter" in out.columns:
            out = out.drop(columns=["Heater Meter"])

        # Merge on player_id when available, else fall back to name.
        if "player_id" in out.columns and "player_id" in heater_map_df.columns:
            tmp = heater_map_df[["player_id", "hotness_1_5"]].copy()
            tmp["player_id"] = tmp["player_id"].astype(str)
            out["player_id"] = out["player_id"].astype(str)
            out = out.merge(tmp, on="player_id", how="left")
        elif "name" in out.columns and "player_name" in heater_map_df.columns:
            tmp = heater_map_df[["player_name", "hotness_1_5"]].copy()
            tmp = tmp.rename(columns={"player_name": "name"})
            out = out.merge(tmp, on="name", how="left")
        else:
            if "Heater Meter" not in out.columns:
                out["Heater Meter"] = None
            return out

        # Convert + drop source column to guarantee unique columns for Streamlit/Arrow.
        if "hotness_1_5" in out.columns:
            out["Heater Meter"] = pd.to_numeric(out["hotness_1_5"], errors="coerce")
            out = out.drop(columns=["hotness_1_5"])
        else:
            out["Heater Meter"] = None

        # Final safety: remove any accidental duplicate columns.
        out = out.loc[:, ~out.columns.duplicated()]
        return out
    except Exception:
        out = df.copy()
        if "Heater Meter" not in out.columns:
            out["Heater Meter"] = None
        return out


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    if not password_gate():
        return

    # persistent slots
    if "sim_results" not in st.session_state:
        st.session_state.sim_results = None
    if "last_tournament_id" not in st.session_state:
        st.session_state.last_tournament_id = None
    if "last_tournament_name" not in st.session_state:
        st.session_state.last_tournament_name = None
    if "last_run_record" not in st.session_state:
        st.session_state.last_run_record = None

    # NEW: seed default slider state keys (so we can programmatically set them)
    if "round_sd" not in st.session_state:
        st.session_state.round_sd = 2.3
    if "wave_r1_gap" not in st.session_state:
        st.session_state.wave_r1_gap = 0.0
    if "wave_r2_gap" not in st.session_state:
        st.session_state.wave_r2_gap = 0.0

    st.title(APP_TITLE)
    st.caption("File-driven weekly simulator + FanDuel lineup builder + saved runs (no API calls).")

    repo_root = Path(__file__).parent

    # =========================
    # RUN HISTORY (LEFT SIDEBAR)
    # =========================
    st.sidebar.header("Run History")
    runs = list_runs(repo_root)

    if runs:
        run_labels = [f"{r.tournament_name} • {r.run_id} • {r.created_utc}" for r in runs]
        sel_idx = st.sidebar.selectbox(
            "Load a past run",
            options=list(range(len(runs))),
            format_func=lambda i: run_labels[i],
            index=0,
        )
        sel_run = runs[sel_idx]

        colh1, colh2 = st.sidebar.columns(2)
        if colh1.button("Load run", use_container_width=True):
            st.session_state.sim_results = load_predictions(sel_run)
            st.session_state.last_tournament_id = sel_run.tournament_id
            st.session_state.last_tournament_name = sel_run.tournament_name
            st.session_state.last_run_record = sel_run
            st.rerun()

        # downloads from sidebar
        with open(sel_run.predictions_path, "rb") as f:
            st.sidebar.download_button(
                "Download predictions.csv",
                f,
                file_name=f"{sel_run.tournament_id}_{sel_run.run_id}_predictions.csv",
                use_container_width=True,
            )
        with open(sel_run.settings_path, "rb") as f:
            st.sidebar.download_button(
                "Download settings.json",
                f,
                file_name=f"{sel_run.tournament_id}_{sel_run.run_id}_settings.json",
                use_container_width=True,
            )

        if sel_run.actuals_path and sel_run.actuals_path.exists():
            with open(sel_run.actuals_path, "rb") as f:
                st.sidebar.download_button(
                    "Download actuals.csv",
                    f,
                    file_name=f"{sel_run.tournament_id}_{sel_run.run_id}_actuals.csv",
                    use_container_width=True,
                )
    else:
        st.sidebar.caption("No saved runs yet. Run a simulation with Auto-save enabled.")

    st.sidebar.divider()

    # =========================
    # WEEKLY DATA LOADING
    # =========================
    st.sidebar.header("Weekly Data")
    weekly_root = repo_root / "data" / "weekly"
    weekly_root.mkdir(parents=True, exist_ok=True)

    week_folders = list_week_folders(weekly_root)

    data_mode = st.sidebar.radio(
        "How to load weekly data?",
        ["Use data/weekly folder", "Upload files (one-off)"],
        index=0,
    )

    weekly_data: WeeklyData | None = None
    week_label = None

    if data_mode == "Use data/weekly folder":
        if not week_folders:
            st.sidebar.warning("No week folders found in data/weekly yet.")
            st.info(
                "Create data/weekly/<week_name>/ and add schedule.json, player_statistics.json, "
                "wgr_rankings.json, plus your FanDuel CSV."
            )
            st.stop()

        week_label = st.sidebar.selectbox("Select week folder", options=week_folders, index=0)
        folder_path = weekly_root / week_label

        csv_choices = list_fanduel_csvs(folder_path)
        if not csv_choices:
            st.sidebar.error("No CSV found in that week folder. Put your FanDuel CSV in it (any name).")
            st.stop()

        fd_choice = st.sidebar.selectbox("FanDuel CSV in folder", csv_choices, index=0)
        weekly_data = load_weekly_data(folder_path, fanduel_filename=fd_choice)

    else:
        st.sidebar.info("Upload the four files for a one-off run.")
        up_sched = st.sidebar.file_uploader("schedule.json", type=["json"])
        up_stats = st.sidebar.file_uploader("player_statistics.json", type=["json"])
        up_wgr = st.sidebar.file_uploader("wgr_rankings.json", type=["json"])
        up_fd = st.sidebar.file_uploader("FanDuel players CSV", type=["csv"])
        up_tee = st.sidebar.file_uploader("tee_times_rd1.json (optional)", type=["json"])

        if up_sched and up_stats and up_wgr and up_fd:
            weekly_data = load_weekly_data(
                folder=None,
                schedule_bytes=up_sched.getvalue(),
                stats_bytes=up_stats.getvalue(),
                wgr_bytes=up_wgr.getvalue(),
                fanduel_bytes=up_fd.getvalue(),
            )
            week_label = "Uploaded files"

    if weekly_data is None:
        st.info("Load a week folder (data/weekly/...) or upload the files to begin.")
        st.stop()

    # =========================
    # OPTIONAL TEE TIMES / WAVE DATA
    # =========================
    tee_times_df = pd.DataFrame()
    tee_times_status = None
    if data_mode == "Use data/weekly folder" and week_label is not None:
        tee_times_path = weekly_root / week_label / "tee_times_rd1.json"
        if tee_times_path.exists():
            try:
                tee_times_df = _load_tee_times_from_path(str(tee_times_path))
                tee_times_status = f"Loaded tee_times_rd1.json ({len(tee_times_df):,} players)."
            except Exception as e:
                tee_times_status = f"Failed to load tee_times_rd1.json: {e}"
    else:
        try:
            if "up_tee" in locals() and up_tee is not None:
                tee_times_df = _load_tee_times_from_bytes(up_tee.getvalue())
                tee_times_status = f"Loaded uploaded tee times ({len(tee_times_df):,} players)."
        except Exception as e:
            tee_times_status = f"Failed to load uploaded tee times: {e}"

    # =========================
    # TOURNAMENT SELECTION
    # =========================
    st.sidebar.header("Tournament")
    tourney_df = weekly_data.schedule_tournaments.copy()
    tourney_df["label"] = tourney_df["name"].astype(str) + " (" + tourney_df["start_date"].astype(str) + ")"
    sel_label = st.sidebar.selectbox("Select tournament", tourney_df["label"].tolist(), index=0)
    sel_row = tourney_df.loc[tourney_df["label"] == sel_label].iloc[0]
    tournament_id = str(sel_row["id"])
    tournament_name = str(sel_row["name"])
    course_meta = weekly_data.get_course_meta(tournament_id)

    # =========================
    # EARLY-SEASON REFERENCE (SIDEBAR)
    # =========================
    st.sidebar.divider()
    st.sidebar.header("Early-season reference")

    ref_path = repo_root / "data" / "reference" / "pga_results_2001-2025.tsv"
    use_ref = st.sidebar.checkbox(
        "Use historical priors (recommended early season)",
        value=True,
        help="Loads historical results and suggests realistic priors (e.g., Round SD).",
    )

    suggested_round_sd = None
    if use_ref:
        if not ref_path.exists():
            st.sidebar.warning("Missing reference TSV.")
            st.sidebar.caption("Add: data/reference/pga_results_2001-2025.tsv")
        else:
            era = st.sidebar.selectbox("Era", ["2015-2025 (recommended)", "2001-2025 (all)"], index=0)
            seasons = (2015, 2025) if era.startswith("2015") else (2001, 2025)

            try:
                ref_df = _load_reference_df(str(ref_path))
                priors = compute_reference_priors(ref_df, seasons=seasons)

                suggested_round_sd = float(priors.suggested_round_sd)

                st.sidebar.metric("Suggested Round SD", f"{priors.suggested_round_sd:.2f}")
                st.sidebar.caption(
                    f"Winner score (median): {priors.winner_score_median:.1f} to par\n\n"
                    f"IQR: {priors.winner_score_iqr[0]:.0f} to {priors.winner_score_iqr[1]:.0f}"
                )

                if st.sidebar.button("Apply suggested Round SD", use_container_width=True):
                    # Updates the slider value (keyed)
                    st.session_state.round_sd = suggested_round_sd
                    st.rerun()

            except Exception as e:
                st.sidebar.error("Failed to load reference priors.")
                st.sidebar.caption(str(e))

    # =========================
    # FIELD + MODEL TABLE
    # =========================
    fd_players = weekly_data.fanduel_players.copy()

    # =========================
    # ROLLING FORM (OPTIONAL)
    # =========================
    use_rolling_sg = st.sidebar.checkbox(
        "Use rolling SG Total form (last 4 weeks)",
        value=False,
        help="If enabled (folder mode only), blends each player's current SG Total with the mean SG Total over the last 4 week folders (including this week).",
    )

    rolling_df = None
    if use_rolling_sg:
        if data_mode != "Use data/weekly folder" or week_label is None:
            st.sidebar.warning("Rolling form requires 'Use data/weekly folder' mode.")
            use_rolling_sg = False
        else:
            try:
                rolling_df = _compute_rolling_sg(str(weekly_root), week_label, n_weeks=4)
                if rolling_df is not None and not rolling_df.empty:
                    used = int(rolling_df.get("rolling_weeks_used", pd.Series([0])).max())
                    st.sidebar.caption(f"Rolling form loaded (up to {used} week(s) available).")
                else:
                    st.sidebar.caption("Rolling form file(s) not found; falling back to current week SG Total.")
            except Exception as e:
                st.sidebar.warning("Failed to compute rolling form; falling back to current week.")
                st.sidebar.caption(str(e))
                rolling_df = None

    st.sidebar.header("Field")
    st.sidebar.caption(f"FanDuel rows: {len(fd_players):,}")
    # Determine dynamic salary bounds (Signature Events can exceed 15,000)
    try:
        salary_max = int(pd.to_numeric(fd_players["Salary"], errors="coerce").max())
        # NaN guard
        if salary_max != salary_max:
            salary_max = 20000
    except Exception:
        salary_max = 20000
    salary_max = max(20000, salary_max)

    min_salary = st.sidebar.slider("Min salary filter", 0, salary_max, 0, step=100)
    max_salary = st.sidebar.slider("Max salary filter", 0, salary_max, salary_max, step=100)
    fd_players = fd_players[(fd_players["Salary"] >= min_salary) & (fd_players["Salary"] <= max_salary)].copy()

    model_table = build_model_table(
        fanduel=fd_players,
        stats=weekly_data.player_stats,
        wgr=weekly_data.wgr_players,
        rolling_sg=rolling_df,
        rolling_weight=0.60,
    )

    if not tee_times_df.empty and "player_id" in model_table.columns and "player_id" in tee_times_df.columns:
        tmp_tt = tee_times_df.copy()
        tmp_tt["player_id"] = tmp_tt["player_id"].astype(str)
        model_table["player_id"] = model_table["player_id"].astype(str)
        merge_cols = [c for c in ["player_id", "tee_time_local_clock", "wave", "starting_hole", "wave_draw_summary"] if c in tmp_tt.columns]
        model_table = model_table.merge(tmp_tt[merge_cols], on="player_id", how="left")

    if model_table.empty:
        st.error("No players matched between FanDuel CSV and your stats/WGR files.")
        st.stop()

    # Pre-compute Heater Meter (hotness 1–5) so we can show it as a column in tables.
    # This is informational only and does NOT alter simulation calculations.
    heater_map_df = pd.DataFrame()
    hr = None
    if data_mode == "Use data/weekly folder" and week_label:
        try:
            hr = _compute_hotness(str(weekly_root), str(week_label))
            if hr is not None and getattr(hr, "form_scores_wide", None) is not None:
                heater_map_df = hr.form_scores_wide[["player_id", "player_name", "hotness_1_5"]].copy()
        except Exception:
            heater_map_df = pd.DataFrame()

    # Attach to the merged field table (display-only)
    model_table = _attach_heater_meter(model_table, heater_map_df)

    # =========================
    # HOTNESS (INFORMATIONAL ONLY)
    # =========================
    with st.expander("Hotness (last 4 weeks) • informational only", expanded=False):
        if data_mode != "Use data/weekly folder" or not week_label:
            st.info("Hotness requires week folders in data/weekly so we can compare the last 4 weeks.")
        else:
            try:
                if hr is None:
                    hr = _compute_hotness(str(weekly_root), str(week_label))
                if not hr.weeks or hr.form_scores_wide.empty:
                    st.warning("No hotness data available.")
                else:
                    hot_df = hr.form_scores_wide.copy()
                    # keep only golfers in this week's FanDuel field
                    field_ids = set(model_table["player_id"].astype(str).tolist())
                    hot_df = hot_df[hot_df["player_id"].astype(str).isin(field_ids)].copy()

                    st.caption(
                        "Weeks used (oldest → newest): " + " → ".join(hr.weeks)
                    )

                    # nicer ordering
                    hot_df = hot_df.sort_values(["hotness_1_5", "hotness_raw"], ascending=[False, False], na_position="last")

                    st.dataframe(
                        hot_df[["player_name", "hotness_1_5", "hotness_raw", "form_last4"]],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "player_name": st.column_config.TextColumn("Golfer"),
                            "hotness_1_5": st.column_config.ProgressColumn(
                                "Hotness (1=colder, 5=hotter)",
                                min_value=1,
                                max_value=5,
                                help="Trend over the last 4 weeks. 5 means improving form; 1 means declining form.",
                            ),
                            "hotness_raw": st.column_config.NumberColumn(
                                "Trend (slope)",
                                format="%.2f",
                                help="Slope of weekly form score (higher = improving).",
                            ),
                            "form_last4": st.column_config.LineChartColumn(
                                "Form sparkline (last 4)",
                                help="Per-week form score (field-relative z-score; higher is better).",
                            ),
                        },
                    )
                    st.caption(
                        "Note: This uses field-relative z-scores per week and inverts metrics where lower is better (e.g., scoring_avg, putt_avg, world_rank/total_driving)."
                    )
            except Exception as e:
                st.error(f"Hotness calculation failed: {e}")

    # =========================
    # SLIDERS
    # =========================
    st.sidebar.header("Course-fit sliders")
    defaults = make_course_fit_weights()

    w_sg_total = st.sidebar.slider("Weight: SG Total", -3.0, 3.0, defaults["sg_total"], 0.05)
    w_sg_t2g = st.sidebar.slider("Weight: SG Tee-to-Green", -3.0, 3.0, defaults["sg_t2g"], 0.05)
    w_putt = st.sidebar.slider("Weight: Putting proxy (strokes_gained)", -3.0, 3.0, defaults["sg_putt_proxy"], 0.05)
    w_birdies = st.sidebar.slider("Weight: Birdies/round", -3.0, 3.0, defaults["birdies_per_round"], 0.05)
    w_gir = st.sidebar.slider("Weight: GIR%", -3.0, 3.0, defaults["gir_pct"], 0.05)
    w_drive = st.sidebar.slider("Weight: Driving distance", -3.0, 3.0, defaults["drive_avg"], 0.05)
    w_acc = st.sidebar.slider("Weight: Driving accuracy", -3.0, 3.0, defaults["drive_acc"], 0.05)
    w_scramble = st.sidebar.slider("Weight: Scrambling%", -3.0, 3.0, defaults["scrambling_pct"], 0.05)
    wgr_weight = st.sidebar.slider("WGR impact (rank → strength)", 0.0, 3.0, 1.0, 0.05)

    st.sidebar.header("Simulation")
    n_sims = st.sidebar.slider("Simulations", 100, 50000, 5000, step=100)
    rng_seed = st.sidebar.text_input("RNG seed (optional)", value="")
    cut_line = st.sidebar.slider("Cut size (after R2)", 50, 80, 65, step=1)

    # CHANGED: Round SD slider uses key so we can apply suggested value
    round_sd = st.sidebar.slider(
        "Round score volatility (stdev)",
        1.0, 4.0,
        float(st.session_state.round_sd),
        0.05,
        key="round_sd",
    )

    course_difficulty = st.sidebar.slider("Course difficulty shift (strokes)", -2.0, 2.0, 0.0, 0.05)

    st.sidebar.header("Weather / Wave")
    use_weather_wave = st.sidebar.checkbox(
        "Enable AM/PM wave adjustment",
        value=False,
        disabled=tee_times_df.empty,
        help="Uses tee_times_rd1.json to identify AM vs PM starters and applies round-specific stroke adjustments.",
    )
    if tee_times_status:
        if tee_times_df.empty:
            st.sidebar.caption(tee_times_status)
        else:
            st.sidebar.success(tee_times_status)
    elif tee_times_df.empty:
        st.sidebar.caption("No tee_times_rd1.json found. Put it in the selected week folder to enable wave adjustments.")

    wave_r1_gap = st.sidebar.slider(
        "Round 1 AM/PM wave gap (strokes)",
        -1.5, 1.5,
        float(st.session_state.wave_r1_gap),
        0.05,
        key="wave_r1_gap",
        disabled=not use_weather_wave,
        help="Positive = AM wave easier in Round 1. Negative = PM wave easier. A value of 0.30 means AM gets -0.15 and PM gets +0.15 so the field average stays neutral.",
    )
    wave_r2_gap = st.sidebar.slider(
        "Round 2 AM/PM wave gap (strokes)",
        -1.5, 1.5,
        float(st.session_state.wave_r2_gap),
        0.05,
        key="wave_r2_gap",
        disabled=not use_weather_wave,
        help="Same concept for Friday. The app assumes players flip waves from Round 1 to Round 2, which is standard for PGA Tour pairings.",
    )

    st.sidebar.header("Run Saving")
    auto_save = st.sidebar.checkbox("Auto-save run outputs", value=True)
    run_note = st.sidebar.text_input("Run note (optional)", value="")

    # =========================
    # HEADER
    # =========================
    colA, colB, colC = st.columns([2.2, 1.2, 1.2])
    with colA:
        st.subheader(tournament_name)
        st.write(f"Selected data: **{week_label}**")
        if course_meta:
            st.caption(
                f"Course: {course_meta.get('course_name', '—')} • "
                f"Par {course_meta.get('par', '—')} • "
                f"Yardage {course_meta.get('yardage', '—')}"
            )
    with colB:
        st.metric("Field size", f"{len(model_table):,}")
    with colC:
        st.metric("Salary cap", _format_money(60000))

    st.markdown("### Field (merged)")
    preview_cols = [
        "player_id", "name", "Heater Meter", "Salary", "FPPG",
        "wgr_rank", "scoring_avg",
        "strokes_gained_total", "strokes_gained_tee_green", "strokes_gained",
        "birdies_per_round", "tee_time_local_clock", "wave",
    ]
    show_cols = [c for c in preview_cols if c in model_table.columns]
    st.dataframe(
        model_table[show_cols],
        use_container_width=True,
        column_config={
            "Heater Meter": _heater_meter_col_config(),
        },
    )

    with st.expander("Weather wave preview", expanded=False):
        if tee_times_df.empty:
            st.info("Load tee_times_rd1.json in the week folder (or upload it in one-off mode) to preview wave assignments.")
        else:
            wave_preview = model_table.copy()
            if use_weather_wave:
                wave_preview = apply_wave_adjustments(wave_preview, r1_wave_gap=float(wave_r1_gap), r2_wave_gap=float(wave_r2_gap))
            cols = [c for c in ["name", "tee_time_local_clock", "wave", "wave_draw_summary", "wave_r1_adjust", "wave_r2_adjust"] if c in wave_preview.columns]
            st.dataframe(wave_preview[cols].sort_values(["wave", "tee_time_local_clock", "name"]), use_container_width=True, hide_index=True)
            st.caption("Positive adjustment = tougher scoring. Negative adjustment = easier scoring.")

    # =========================
    # RUN SIMULATION
    # =========================
    st.markdown("## Tournament simulation")

    if st.button("Run simulation", type="primary", use_container_width=True):
        weights = {
            "sg_total": w_sg_total,
            "sg_t2g": w_sg_t2g,
            "sg_putt_proxy": w_putt,
            "birdies_per_round": w_birdies,
            "gir_pct": w_gir,
            "drive_avg": w_drive,
            "drive_acc": w_acc,
            "scrambling_pct": w_scramble,
        }

        sim_input = model_table.copy()
        if use_weather_wave and not tee_times_df.empty:
            sim_input = apply_wave_adjustments(sim_input, r1_wave_gap=float(wave_r1_gap), r2_wave_gap=float(wave_r2_gap))

        cfg = SimConfig(
            n_sims=int(n_sims),
            rng_seed=_safe_int(rng_seed, default=None) if rng_seed.strip() else None,
            cut_size=int(cut_line),
            round_sd=float(round_sd),
            course_difficulty=float(course_difficulty),
            wgr_weight=float(wgr_weight),
            course_fit_weights=weights,
        )

        with st.spinner("Simulating..."):
            results = simulate_tournament(sim_input, cfg)

        st.session_state.sim_results = results
        st.session_state.last_tournament_id = tournament_id
        st.session_state.last_tournament_name = tournament_name

        st.success("Simulation complete.")

        if auto_save:
            settings_payload = {
                "week_label": week_label,
                "tournament_id": tournament_id,
                "tournament_name": tournament_name,
                "run_note": run_note,
                "sim": {
                    "n_sims": int(n_sims),
                    "rng_seed": _safe_int(rng_seed, default=None) if rng_seed.strip() else None,
                    "cut_size": int(cut_line),
                    "round_sd": float(round_sd),
                    "course_difficulty": float(course_difficulty),
                    "wgr_weight": float(wgr_weight),
                },
                "course_fit_weights": weights,
                "field_filters": {
                    "min_salary": int(min_salary),
                    "max_salary": int(max_salary),
                },
                "weather_wave": {
                    "enabled": bool(use_weather_wave and not tee_times_df.empty),
                    "tee_times_file_present": bool(not tee_times_df.empty),
                    "round_1_gap": float(wave_r1_gap) if use_weather_wave and not tee_times_df.empty else 0.0,
                    "round_2_gap": float(wave_r2_gap) if use_weather_wave and not tee_times_df.empty else 0.0,
                },
                # NEW: record reference context (if used)
                "reference": {
                    "enabled": bool(use_ref),
                    "ref_file_present": bool(ref_path.exists()),
                    "suggested_round_sd": suggested_round_sd,
                },
            }

            rec = save_run(
                repo_root=repo_root,
                tournament_id=tournament_id,
                tournament_name=tournament_name,
                settings=settings_payload,
                predictions=results,
            )
            st.session_state.last_run_record = rec
            st.info(f"Saved run: {rec.run_id}")

    # =========================
    # SHOW LATEST RESULTS + DOWNLOADS + ACTUALS PLACEHOLDER
    # =========================
    results = _attach_heater_meter(st.session_state.sim_results, heater_map_df)
    st.session_state.sim_results = results
    if results is not None and not results.empty:
        st.markdown("### Latest simulation results")
        summ_cols = [
            "name", "Heater Meter", "Salary", "FPPG", "tee_time_local_clock", "wave",
            "win_pct", "top10_pct", "make_cut_pct", "avg_finish", "proj_fd_points",
        ]
        summ_cols = [c for c in summ_cols if c in results.columns]
        res_col_config = {}
        if "Heater Meter" in summ_cols:
            res_col_config["Heater Meter"] = _heater_meter_col_config()
        st.dataframe(
            results[summ_cols].head(50),
            use_container_width=True,
            column_config=res_col_config if res_col_config else None,
        )

        st.download_button(
            "Download current predictions CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name=f"predictions_{st.session_state.last_tournament_id}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # If last run was saved, show one-click downloads for that run
        rec = st.session_state.last_run_record
        if rec is not None:
            st.markdown("### Saved run files")
            c1, c2, c3 = st.columns(3)
            with open(rec.predictions_path, "rb") as f:
                c1.download_button(
                    "Download saved predictions.csv",
                    f,
                    file_name=f"{rec.tournament_id}_{rec.run_id}_predictions.csv",
                    use_container_width=True,
                )
            with open(rec.settings_path, "rb") as f:
                c2.download_button(
                    "Download saved settings.json",
                    f,
                    file_name=f"{rec.tournament_id}_{rec.run_id}_settings.json",
                    use_container_width=True,
                )

            # Placeholder: upload actuals later
            st.markdown("### Import actual results (placeholder)")
            st.caption("After the tournament ends, upload a CSV here and it will be saved next to this run as actuals.csv.")
            up_actuals = st.file_uploader("Upload actuals.csv for this run", type=["csv"], key="actuals_uploader")
            if up_actuals is not None:
                path = save_actuals_csv(rec, up_actuals.getvalue())
                st.success(f"Saved actuals to: {path.as_posix()}")

                with open(path, "rb") as f:
                    c3.download_button(
                        "Download actuals.csv",
                        f,
                        file_name=f"{rec.tournament_id}_{rec.run_id}_actuals.csv",
                        use_container_width=True,
                    )

    # =========================
    # LINEUP BUILDER
    # =========================
    st.markdown("## FanDuel lineup builder (6 golfers, ≤ $60,000)")

    if results is None or results.empty:
        st.info("Run a simulation first (or load a past run from Run History).")
        st.stop()

    col1, col2, col3 = st.columns([1.2, 1.2, 1.6])
    with col1:
        candidate_k = st.number_input("Candidate pool size (search space)", 12, 120, 40, 1)
    with col2:
        lock_names = st.multiselect("Lock players (optional)", results["name"].tolist(), default=[])
    with col3:
        exclude_names = st.multiselect("Exclude players (optional)", results["name"].tolist(), default=[])

    blend_alpha = st.slider(
        "Lineup scoring weight (Simulation vs FanDuel FPPG)",
        0.0, 1.0, 0.75, 0.05,
        help="0.0 = only FanDuel FPPG, 1.0 = only sim projection. Middle values blend both."
    )

    if st.button("Build best lineup", use_container_width=True):
        with st.spinner("Searching best lineup under $60,000..."):
            lineup, meta = optimize_fanduel_lineup(
                sim_results=results,
                salary_cap=60000,
                lineup_size=6,
                candidate_pool=int(candidate_k),
                lock_names=set(lock_names),
                exclude_names=set(exclude_names),
                blend_alpha=float(blend_alpha),
            )

        if lineup is None or lineup.empty:
            st.error("No valid lineup found. Try increasing candidate pool or removing locks.")
        else:
            st.success("Lineup found.")
            # Ensure Heater Meter is present for display
            lineup = _attach_heater_meter(lineup, heater_map_df)

            cols = ["name", "Heater Meter", "Salary", "FPPG", "proj_fd_points"]
            if "blend_points" in lineup.columns:
                cols.append("blend_points")
            cols += ["win_pct", "top10_pct", "make_cut_pct"]
            cols = [c for c in cols if c in lineup.columns]
            lu_col_config = {}
            if "Heater Meter" in cols:
                lu_col_config["Heater Meter"] = _heater_meter_col_config()
            st.dataframe(
                lineup[cols],
                use_container_width=True,
                column_config=lu_col_config if lu_col_config else None,
            )

            st.metric("Total salary", _format_money(meta.get("total_salary", lineup["Salary"].sum())))
            st.metric("Projected lineup score", f"{meta.get('total_points', 0.0):.2f}")

            st.download_button(
                "Download lineup CSV",
                data=lineup.to_csv(index=False).encode("utf-8"),
                file_name=f"fanduel_lineup_{st.session_state.last_tournament_id}.csv",
                mime="text/csv",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
