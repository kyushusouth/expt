import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm


def generate_data(
    seed: int,
    n_source: int,
    n_target: int,
    n_numerical_feature: int,
    n_categorical_feature: int,
    n_category: int,
    shift: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """source と target のダミーデータを生成する。

    Args:
        seed: シード値
        n_source: ソース分布のサンプル数
        n_target: ターゲット分布のサンプル数
        n_numerical_feature: 数値特徴量の数
        n_categorical_feature: カテゴリ特徴量の数
        n_category: カテゴリ特徴量のカテゴリ数
        shift: 分布のズレを制御するパラメータ

    Returns:
        (
            ソース分布のサンプル,
            ターゲット分布のサンプル,
            数値特徴量名のリスト,
            カテゴリ特徴量名のリスト,
        )
    """
    rng = np.random.default_rng(seed)

    numerical_feature_names = []
    categorical_feature_names = []
    source: dict[str, np.ndarray] = {}
    target: dict[str, np.ndarray] = {}

    # 連続特徴量
    for i in range(n_numerical_feature):
        loc = rng.uniform(-2.0, 2.0)
        scale = rng.uniform(0.5, 2.0)
        source[f"num_{i}"] = rng.normal(loc, scale, n_source)
        target[f"num_{i}"] = rng.normal(loc + shift * scale, scale, n_target)
        numerical_feature_names.append(f"num_{i}")

    # カテゴリ特徴量
    categories = np.arange(n_categorical_feature)
    for i in range(n_category):
        base_probs = np.full(n_categorical_feature, 1.0 / n_categorical_feature)
        bias = rng.random(n_categorical_feature)
        bias /= bias.sum()
        probs = base_probs + shift * (bias - base_probs)
        probs = np.clip(probs, a_min=1e-6, a_max=None)
        shifted_probs = probs / probs.sum()
        source[f"cat_{i}"] = rng.choice(categories, n_source, p=base_probs)
        target[f"cat_{i}"] = rng.choice(categories, n_target, p=shifted_probs)
        categorical_feature_names.append(f"cat_{i}")

    return (
        pd.DataFrame(source),
        pd.DataFrame(target),
        numerical_feature_names,
        categorical_feature_names,
    )


def detect_drift(
    source: pd.DataFrame,
    target: pd.DataFrame,
    numerical_feature_names: list[str],
    categorical_feature_names: list[str],
    alpha: float,
) -> tuple[bool, pd.DataFrame]:
    """source/target 間のデータドリフトを検知する

    Args:
        source: ソース分布のサンプル
        target: ターゲット分布のサンプル
        numerical_feature_names: 数値特徴量名のリスト
        categorical_feature_names: カテゴリ特徴量名のリスト
        alpha: 有意水準

    Returns:
        (判定結果, 特徴量ごとの判定結果)
    """
    rows: list[dict[str, object]] = []

    for col in source.columns:
        src = source[col].to_numpy()
        tgt = target[col].to_numpy()

        if col in numerical_feature_names:
            stat, p = stats.ks_2samp(src, tgt, alternative="two-sided", method="auto")
            test = "ks"
        elif col in categorical_feature_names:
            categories = np.unique(np.concatenate([src, tgt]))
            src_counts = np.array([(src == c).sum() for c in categories], dtype=float)
            tgt_counts = np.array([(tgt == c).sum() for c in categories], dtype=float)
            table = np.vstack([src_counts, tgt_counts])
            stat, p, _, _ = stats.chi2_contingency(table)
            test = "chi2"
        else:
            raise ValueError(f"{col} is not included.")

        rows.append({"feature": col, "test": test, "statistic": stat, "p_value": p})

    per_feature = pd.DataFrame(rows)

    n_tests = len(per_feature)

    per_feature["p_value_corrected"] = (per_feature["p_value"] * n_tests).clip(
        upper=1.0
    )
    per_feature["drifted"] = per_feature["p_value_corrected"] < alpha
    is_drifted = bool(per_feature["drifted"].any())

    return is_drifted, per_feature


def main() -> None:
    # パラメータ
    n_source = 10000
    n_target = 10000
    n_numerical_feature = 100
    n_categorical_feature = 20
    n_category = 4
    # shift = 0.0
    alpha = 0.05
    n_sim = 1000

    """
    shift=0.0, positive_rate = 0.043
    shift=0.01, positive_rate = 0.186
    shift=0.03, positive_rate = 0.989
    shift=0.05, positive_rate = 1.0
    """

    # 単発のテスト
    # source, target, numerical_feature_names, categorical_feature_names = generate_data(
    #     seed=42,
    #     n_source=n_source,
    #     n_target=n_target,
    #     n_numerical_feature=n_numerical_feature,
    #     n_categorical_feature=n_categorical_feature,
    #     n_category=n_category,
    #     shift=shift,
    # )
    # is_drifted, result_per_feature = detect_drift(
    #     source, target, numerical_feature_names, categorical_feature_names, alpha
    # )
    # print(result_per_feature.to_markdown())
    # print(f"is_drifted = {is_drifted}")

    # シミュレーション
    for shift in [0.0, 0.01, 0.03, 0.05]:
        n_pos = 0
        for seed in tqdm(range(n_sim)):
            source, target, numerical_feature_names, categorical_feature_names = (
                generate_data(
                    seed=seed,
                    n_source=n_source,
                    n_target=n_target,
                    n_numerical_feature=n_numerical_feature,
                    n_categorical_feature=n_categorical_feature,
                    n_category=n_category,
                    shift=shift,
                )
            )

            is_drifted, result_per_feature = detect_drift(
                source,
                target,
                numerical_feature_names,
                categorical_feature_names,
                alpha,
            )
            n_pos += 1 if is_drifted else 0

        positive_rate = n_pos / n_sim
        print(f"shift={shift}, positive_rate = {positive_rate}")


if __name__ == "__main__":
    main()
