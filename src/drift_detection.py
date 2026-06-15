"""データドリフト検知のデモ。

連続特徴量は Kolmogorov-Smirnov 検定、カテゴリ特徴量はカイ二乗検定で
source と target の分布差を評価する。得られた p 値群をボンフェローニ補正し、
補正後 p 値が有意水準未満の特徴量が 1 つでもあれば「ドリフトあり」と判定する。

サンプル数・分布のズレ具合をパラメータで変えながら、検知器の精度
(真陽性率・偽陽性率)をシミュレーションで評価できる。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

# ----------------------------------------------------------------------------
# データ生成
# ----------------------------------------------------------------------------


@dataclass
class DataConfig:
    """ダミーデータ生成の設定。

    Attributes:
        n_source: source のサンプル数。
        n_target: target のサンプル数。
        shift: 分布の離れ具合 (0 でドリフトなし)。
            - 連続特徴量: 平均を `shift * 標準偏差` だけずらす。
            - カテゴリ特徴量: カテゴリ確率を `shift` に比例して偏らせる。
        n_numeric: 連続特徴量の数。
        n_categorical: カテゴリ特徴量の数。
        n_categories: 各カテゴリ特徴量の水準数。
        seed: 乱数シード。
    """

    n_source: int = 1000
    n_target: int = 1000
    shift: float = 0.0
    n_numeric: int = 100
    n_categorical: int = 20
    n_categories: int = 4
    seed: int | None = None


def _make_categorical_probs(
    n_categories: int, shift: float, rng: np.random.Generator
) -> np.ndarray:
    """カテゴリの基準確率を、shift に応じて偏らせた確率に変換する。"""
    base = np.full(n_categories, 1.0 / n_categories)
    if shift == 0.0:
        return base
    # shift が大きいほど特定カテゴリへ確率を寄せる
    bias = rng.random(n_categories)
    bias /= bias.sum()
    probs = base + shift * (bias - base)
    probs = np.clip(probs, 1e-6, None)
    return probs / probs.sum()


def generate_data(config: DataConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """source と target のダミーデータを生成する。

    Returns:
        (source_df, target_df)。列名は num_0.. / cat_0.. 。
    """
    rng = np.random.default_rng(config.seed)

    source: dict[str, np.ndarray] = {}
    target: dict[str, np.ndarray] = {}

    # 連続特徴量: target は平均を shift*sigma ずらす
    for i in range(config.n_numeric):
        loc = rng.uniform(-2.0, 2.0)
        scale = rng.uniform(0.5, 2.0)
        source[f"num_{i}"] = rng.normal(loc, scale, config.n_source)
        target[f"num_{i}"] = rng.normal(
            loc + config.shift * scale, scale, config.n_target
        )

    # カテゴリ特徴量: target はカテゴリ確率を偏らせる
    categories = np.arange(config.n_categories)
    for i in range(config.n_categorical):
        base = np.full(config.n_categories, 1.0 / config.n_categories)
        shifted = _make_categorical_probs(config.n_categories, config.shift, rng)
        source[f"cat_{i}"] = rng.choice(categories, config.n_source, p=base)
        target[f"cat_{i}"] = rng.choice(categories, config.n_target, p=shifted)

    return pd.DataFrame(source), pd.DataFrame(target)


# ----------------------------------------------------------------------------
# ドリフト検定
# ----------------------------------------------------------------------------


@dataclass
class DriftResult:
    """ドリフト検知の結果。"""

    drifted: bool
    alpha: float
    alpha_corrected: float
    per_feature: pd.DataFrame = field(default_factory=pd.DataFrame)


def detect_drift(
    source: pd.DataFrame,
    target: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> DriftResult:
    """source/target 間のデータドリフトを検知する。

    連続列(数値型)は KS 検定、それ以外はカイ二乗検定で p 値を求め、
    全特徴量にわたってボンフェローニ補正(alpha / 検定数)を適用する。
    補正後 alpha 未満の p 値が 1 つでもあればドリフトありと判定。
    """
    rows: list[dict[str, object]] = []

    for col in source.columns:
        src = source[col].to_numpy()
        tgt = target[col].to_numpy()
        if pd.api.types.is_float_dtype(source[col]):
            stat, p = stats.ks_2samp(src, tgt, alternative="two-sided", method="auto")
            test = "ks"
        else:
            categories = np.unique(np.concatenate([src, tgt]))
            src_counts = np.array([(src == c).sum() for c in categories], dtype=float)
            tgt_counts = np.array([(tgt == c).sum() for c in categories], dtype=float)
            table = np.vstack([src_counts, tgt_counts])
            stat, p, _, _ = stats.chi2_contingency(table)
            test = "chi2"

        rows.append({"feature": col, "test": test, "statistic": stat, "p_value": p})

    per_feature = pd.DataFrame(rows)

    n_tests = len(per_feature)
    alpha_corrected = alpha / n_tests if n_tests else alpha

    # ボンフェローニ補正は「p値を n 倍」または「閾値を 1/n」のどちらでも等価
    per_feature["p_value_corrected"] = (per_feature["p_value"] * n_tests).clip(
        upper=1.0
    )
    per_feature["drifted"] = per_feature["p_value"] < alpha_corrected

    return DriftResult(
        drifted=bool(per_feature["drifted"].any()),
        alpha=alpha,
        alpha_corrected=alpha_corrected,
        per_feature=per_feature,
    )


# ----------------------------------------------------------------------------
# 精度評価シミュレーション
# ----------------------------------------------------------------------------


def evaluate(
    *,
    n_source: int = 1000,
    n_target: int = 1000,
    shift: float = 1.0,
    alpha: float = 0.05,
    n_trials: int = 200,
    base_seed: int = 0,
) -> dict[str, float]:
    """検知器の精度をモンテカルロで評価する。

    各試行で
      - shift=0 のデータ(ドリフトなし)→ 検知したら偽陽性
      - shift=shift のデータ(ドリフトあり)→ 検知できたら真陽性
    を生成し、検出力 (TPR) と偽陽性率 (FPR) を集計する。
    """
    true_positives = 0
    false_positives = 0

    for t in tqdm(range(n_trials)):
        null_cfg = DataConfig(
            n_source=n_source,
            n_target=n_target,
            shift=0.0,
            seed=base_seed + t,
        )
        drift_cfg = DataConfig(
            n_source=n_source,
            n_target=n_target,
            shift=shift,
            seed=base_seed + 10000 + t,
        )

        s0, t0 = generate_data(null_cfg)
        if detect_drift(s0, t0, alpha=alpha).drifted:
            false_positives += 1

        s1, t1 = generate_data(drift_cfg)
        if detect_drift(s1, t1, alpha=alpha).drifted:
            true_positives += 1

    return {
        "shift": shift,
        "n_source": n_source,
        "n_target": n_target,
        "power_tpr": true_positives / n_trials,
        "false_positive_rate": false_positives / n_trials,
        "n_trials": n_trials,
    }


# ----------------------------------------------------------------------------
# デモ実行
# ----------------------------------------------------------------------------


def _demo() -> None:
    print("=== 単発のドリフト検知 (shift=1.0) ===")
    cfg = DataConfig(n_source=10000, n_target=10000, shift=0.1, seed=42)
    source, target = generate_data(cfg)
    result = detect_drift(source, target, alpha=0.05)
    print(f"補正前 alpha=0.05 / 補正後 alpha={result.alpha_corrected:.4f}")
    print(result.per_feature.to_string(index=False))
    print(f"=> ドリフト判定: {result.drifted}\n")

    print("=== 精度評価: shift・サンプル数を振る ===")
    rows = []
    for n in [10000]:
        for shift in (0.0, 0.05, 0.1):
            print(f"n: {n}, shift: {shift}")
            rows.append(evaluate(n_source=n, n_target=n, shift=shift, n_trials=100))
    summary = pd.DataFrame(rows)[
        ["n_source", "shift", "power_tpr", "false_positive_rate"]
    ]
    print(summary.to_string(index=False))


if __name__ == "__main__":
    _demo()
