
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

# === 設定 ===
THRESHOLD = 0.19
MODEL_PATH = "Model/M4_catboost_raw_model.cbm"
X_PATH = "predict_dataset_procced/X_static_catboost_predict.csv"
OUT_PROBA = "p_catboost_predict.npy"
OUT_BINARY = "y_catboost_predict.npy"

# === 預測流程 ===
print("載入資料與模型...")
X = pd.read_csv(X_PATH)
model = CatBoostClassifier()
model.load_model(MODEL_PATH)

print("開始預測機率...")
probs = model.predict_proba(X)[:, 1]
preds = (probs > THRESHOLD).astype(int)

np.save(OUT_PROBA, probs)
np.save(OUT_BINARY, preds)

print(f"預測機率已儲存至：{OUT_PROBA}, shape = {probs.shape}")
print(f"二值化預測已儲存至：{OUT_BINARY}, shape = {preds.shape}")
print(f"使用 threshold = {THRESHOLD}")
