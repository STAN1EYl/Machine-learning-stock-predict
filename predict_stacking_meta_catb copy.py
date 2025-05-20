# predict_stacking_meta_catb.py
# 使用 CatBoost 融合器對預測資料進行分類，並產出 submission

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

# === 載入 base model 預測機率 ===
p_catb = np.load("stock_predict_5/p_catb_predict.npy")
p_xgb = np.load("stock_predict_5/p_xgb_predict.npy")
X_meta = np.vstack([p_catb, p_xgb]).T

# === 載入訓練好的 CatBoost meta model ===
model = CatBoostClassifier()
model.load_model("stock_predict_5/meta_model_catb.cbm")

# === 預測機率 ===
p_stack = model.predict_proba(X_meta)[:, 1]
np.save("stock_predict_5/p_stack_predict_catb.npy", p_stack)

# === 使用最佳 threshold===
best_threshold = 0.2971#0.1189
labels = (p_stack >= best_threshold).astype(int)


df = pd.read_csv("stock_predict_5/submission_template_public.csv")
df["飆股"] = labels

df.to_csv("stock_predict_5/submission_final_catb.csv", index=False)
print("✅ 已產出 submission_final_catb.csv（使用 CatBoost stacking）")
