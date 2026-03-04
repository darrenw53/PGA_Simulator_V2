import pandas as pd

def calibrate_global(sim_df: pd.DataFrame, actual_df: pd.DataFrame) -> dict:
    m = sim_df.merge(actual_df, on="player_id", how="inner")
    if m.empty:
        return {"note": "No overlap between sim and actual."}

    corr = float(m["avg_finish"].corr(m["actual_finish"]))
    err = float((m["avg_finish"] - m["actual_finish"]).mean())

    return {
        "overlap_n": int(len(m)),
        "pred_vs_actual_finish_corr": corr,
        "mean_finish_error": err,
    }
