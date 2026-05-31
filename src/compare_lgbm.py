import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score


def generate_dummy_ranking_data(num_queries=100, items_per_query=20, seed=42):
    np.random.seed(seed)

    data = []

    for qid in range(num_queries):
        user_preference_vec = np.random.rand(4)
        positions = np.arange(1, items_per_query + 1)

        for pos in positions:
            item_vec = np.random.rand(4)

            query_feat = user_preference_vec * np.random.uniform(0.5, 1.5, 4)
            item_feat = item_vec * np.random.uniform(0.5, 1.5, 4)

            user_total_clicks = np.random.choice(
                [0, 5, 10, 50, 100], p=[0.1, 0.2, 0.3, 0.3, 0.1]
            )
            item_clicks_in_category = (
                np.random.randint(0, int(user_total_clicks) + 1)
                if user_total_clicks > 0
                else 0
            )

            true_relevance_score = np.dot(user_preference_vec, item_vec)

            bias_factor = 1.0 / np.sqrt(pos)
            observed_score = true_relevance_score * bias_factor + np.random.normal(
                0, 0.1
            )

            if observed_score > 0.6:
                label = 2
            elif observed_score > 0.3:
                label = 1
            else:
                label = 0

            row = {
                "query_id": qid,
                "position": pos,
                "user_total_clicks": user_total_clicks,
                "item_clicks_in_category": item_clicks_in_category,
                "label": label,
            }

            for i in range(4):
                row[f"q_vec_{i}"] = query_feat[i]
                row[f"i_vec_{i}"] = item_feat[i]

            data.append(row)

    return pd.DataFrame(data)


def engineer_features(df, is_train=True):
    df_feat = df.copy()

    df_feat["vec_diff_0"] = df_feat["q_vec_0"] - df_feat["i_vec_0"]

    df_feat["click_ratio"] = df_feat.apply(
        lambda row: (
            row["item_clicks_in_category"] / row["user_total_clicks"]
            if row["user_total_clicks"] > 0
            else np.nan
        ),
        axis=1,
    )

    q_cols = [f"q_vec_{i}" for i in range(4)]
    i_cols = [f"i_vec_{i}" for i in range(4)]

    df_feat["dot_product"] = (df_feat[q_cols].values * df_feat[i_cols].values).sum(
        axis=1
    )

    q_norm = np.linalg.norm(df_feat[q_cols].values, axis=1)
    i_norm = np.linalg.norm(df_feat[i_cols].values, axis=1)
    q_norm = np.where(q_norm == 0, 1e-9, q_norm)
    i_norm = np.where(i_norm == 0, 1e-9, i_norm)

    df_feat["cosine_similarity"] = df_feat["dot_product"] / (q_norm * i_norm)

    return df_feat


def main():
    use_interaction_features = True
    use_dot_product = True
    
    df_train = generate_dummy_ranking_data(num_queries=150, seed=42)
    df_test = generate_dummy_ranking_data(num_queries=50, seed=99)

    df_train_feat = engineer_features(df_train)
    df_test_feat = engineer_features(df_test)

    train_groups = df_train_feat.groupby("query_id").size().values
    test_groups = df_test_feat.groupby("query_id").size().values

    features = (
        [
            "user_total_clicks",
            "item_clicks_in_category",
            "cosine_similarity",
        ]
        + [f"q_vec_{i}" for i in range(4)]
        + [f"i_vec_{i}" for i in range(4)]
    )
    if use_interaction_features:
        features += ["vec_diff_0", "click_ratio"]
    if use_dot_product:
        features += ["dot_product"]

    X_train = df_train_feat[features]
    y_train = df_train_feat["label"]
    pos_train = df_train_feat["position"]
    X_test = df_test_feat[features]
    y_test = df_test_feat["label"]
    pos_test = df_test_feat["position"]

    train_set = lgb.Dataset(
        X_train, label=y_train, group=train_groups, position=pos_train
    )
    test_set = lgb.Dataset(X_test, label=y_test, group=test_groups, position=pos_test)
    
    params = {
        "num_leaves": 31,
        "learning_rate": 0.01,
        "n_estimators": 100,
        "objective": "lambdarank",
        "random_state": 42,
        "max_bin": 255,
    }
    lgb.train(
        params=params,
        train_set=train_set,
    )


if __name__ == "__main__":
    main()
