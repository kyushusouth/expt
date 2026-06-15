import numpy as np
import pandas as pd
import streamlit as st

# 1. データの準備（実際はBQから取得）
# data = pd.read_gbq("SELECT model_name, metric_name, value FROM ...")
raw_data = pd.DataFrame([
    {"model": "Baseline", "NDCG": 0.421, "CTR": 0.035, "IPS": 0.015},
    {"model": "Candidate_A", "NDCG": 0.435, "CTR": 0.038, "IPS": 0.012},
    {"model": "Candidate_B", "NDCG": 0.440, "CTR": 0.032, "IPS": 0.018},
])

# 2. ピボット処理：モデルを列に、指標を行にする
df_pivoted = raw_data.set_index("model").T

# 3. 改善率（Lift）の計算
# ベースライン列を取得
baseline_col = df_pivoted["Baseline"]

# 全列に対してベースライン比（%）を計算する新しいDataFrameを作成
lift_df = df_pivoted.div(baseline_col, axis=0) * 100

# 4. スタイリング（カラースケール）の定義
def style_lift(styler):
    styler.background_gradient(
        cmap="RdYlGn",   # 赤(低)-黄(中)-緑(高)のグラデーション
        axis=1, 
        low=0.5,         # スケールの調整
        high=0.5,
        vmin=90,         # 90%以下は濃い赤
        vmax=110         # 110%以上は濃い緑
    )
    styler.format("{:.1f}%") # 表示をパーセント形式に
    return styler

st.subheader("実験評価グリッド（改善率）")
# Streamlitで表示
st.dataframe(style_lift(lift_df.style))
