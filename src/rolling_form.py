import os
import json
import numpy as np
import pandas as pd


def calculate_rolling_sg(weekly_path):

    weeks = sorted(os.listdir(weekly_path))
    rows = []

    for week_index, week in enumerate(weeks):

        stats_file = os.path.join(
            weekly_path,
            week,
            "player_statistics.json"
        )

        if not os.path.exists(stats_file):
            continue

        with open(stats_file) as f:
            stats = json.load(f)

        for p in stats:

            rows.append({
                "player_name": p.get("player_name"),
                "week_index": week_index,
                "sg_total": p.get("strokes_gained_total", 0),
                "sg_t2g": p.get("strokes_gained_tee_green", 0)
            })

    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame()

    # recency weighting
    df["weight"] = np.exp(df["week_index"] / df["week_index"].max())

    rolling = df.groupby("player_name").apply(
        lambda x: pd.Series({
            "rolling_sg_total": np.average(
                x["sg_total"], weights=x["weight"]
            ),
            "rolling_sg_t2g": np.average(
                x["sg_t2g"], weights=x["weight"]
            )
        })
    )

    rolling.reset_index(inplace=True)

    return rolling
