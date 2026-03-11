import numpy as np
import pandas as pd

def run_simulation(players, n_sims=10000, round_sd=3.2, course_difficulty=1.0):

    rng = np.random.default_rng()

    df = players.copy()

    # --- Fix elite flattening ---
    # compress extremely high round volatility
    effective_round_sd = min(round_sd, 3.0)

    strengths = df["skill"].values

    results = []

    for _ in range(n_sims):

        scores = []

        for i, strength in enumerate(strengths):

            # stronger players get slightly lower variance
            player_sd = effective_round_sd * (1 - min(strength * 0.03, 0.15))

            mu = 72 - (0.6 * strength) + course_difficulty

            rounds = rng.normal(
                loc=mu,
                scale=player_sd,
                size=4
            )

            total = rounds.sum()

            scores.append(total)

        scores = np.array(scores)

        ranks = scores.argsort().argsort()

        results.append(ranks)

    results = np.array(results)

    df["win_pct"] = (results == 0).mean(axis=0)
    df["top5_pct"] = (results <= 4).mean(axis=0)
    df["top10_pct"] = (results <= 9).mean(axis=0)

    return df
