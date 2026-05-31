from functools import partial

import numpy as np
import optuna
import pandas as pd

np.random.seed(42)


def generate_search_logs(
    rng: np.random.Generator, n_sessions: int, items_per_session: int
):
    scenes = ["飲み会", "ご飯", "カフェ"]
    data = []

    for session_id in range(n_sessions):
        query_scene = rng.choice(scenes)

        for item_idx in range(items_per_session):
            score_nomikai = rng.integers(1, 11)
            score_gohan = rng.integers(1, 11)
            score_cafe = rng.integers(1, 11)

            if query_scene == "飲み会":
                target_score = score_nomikai
            elif query_scene == "ご飯":
                target_score = score_gohan
            else:
                target_score = score_cafe

            cv_prob = 1 / (1 + np.exp(-(target_score - 3.2) * 1.5))
            is_cv = rng.binomial(1, cv_prob)

            data.append(
                {
                    "session_id": session_id,
                    "query_scene": query_scene,
                    "score_nomikai": score_nomikai,
                    "score_gohan": score_gohan,
                    "score_cafe": score_cafe,
                    "is_cv": is_cv,
                }
            )

    return pd.DataFrame(data)


def objective(trial: optuna.Trial, df_logs: pd.DataFrame):
    t_nomikai = trial.suggest_int("t_nomikai", 1, 10)
    t_gohan = trial.suggest_int("t_gohan", 1, 10)
    t_cafe = trial.suggest_int("t_cafe", 1, 10)

    cond_nomikai = (df_logs["query_scene"] == "飲み会") & (
        df_logs["score_nomikai"] >= t_nomikai
    )
    cond_gohan = (df_logs["query_scene"] == "ご飯") & (
        df_logs["score_gohan"] >= t_gohan
    )
    cond_cafe = (df_logs["query_scene"] == "カフェ") & (df_logs["score_cafe"] >= t_cafe)

    true_cv_counts = df_logs["is_cv"].sum()

    pred_row_counts = df_logs.loc[cond_nomikai | cond_gohan | cond_cafe].shape[0]
    pred_cv_counts = df_logs.loc[cond_nomikai | cond_gohan | cond_cafe, "is_cv"].sum()

    precision = pred_cv_counts / pred_row_counts
    recall = pred_cv_counts / true_cv_counts

    return precision, recall


def main():
    rng = np.random.default_rng(42)

    study = optuna.create_study(
        directions=["maximize", "maximize"],
        sampler=optuna.samplers.NSGAIISampler(seed=42),
    )

    df_logs = generate_search_logs(rng=rng, n_sessions=10000, items_per_session=50)

    study.optimize(partial(objective, df_logs=df_logs), n_trials=300)
    fig = optuna.visualization.plot_pareto_front(study, include_dominated_trials=True)
    fig.write_html("pareto_front.html")


if __name__ == "__main__":
    main()
