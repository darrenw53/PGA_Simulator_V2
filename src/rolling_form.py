import os
import json
import numpy as np
import pandas as pd

def calculate_rolling_sg(weekly_path):

    weeks = sorted(os.listdir(weekly_path))

    data = []

    for i, wk in enumerate(weeks):

        stats_file = os.path.join(
            weekly_path,
            wk,
            "player_statistics.json"
        )

        if not os.path.exists(stats_file):
            continue

        with open(stats_file) as f:
            stats = json.load(f)

        for p in stats:

            data.append({
                "player": p["player_name"],
                "week": i,
                "sg_total": p.get("strokes_gained_total", 0),
                "sg_t2g": p.get("strokes_gained_tee_green", 0)
            })

    df = pd.DataFrame(data)

    weights = np.exp(df["week"] / df["week"].max())

    df["w"] = weights

    roll = df.groupby("player").apply(
        lambda x: pd.Series({
            "rolling_sg_total":
                np.average(x.sg_total, weights=x.w),
            "rolling_sg_t2g":
                np.average(x.sg_t2g, weights=x.w)
        })
    )

    roll.reset_index(inplace=True)

    return roll
