
import numpy as np
import joblib

THRESHOLD = 0.89
MODEL_PATH = "Model/M3_xgb_raw_model.pkl"
X_PATH = "predict_dataset_procced/X_full_predict.npy"
OUT_PATH_BIN = "y_xgb_predict.npy"
OUT_PATH_PROBA = "p_xgb_predict.npy"

print("載入預測資料與模型...")
X = np.load(X_PATH)
model = joblib.load(MODEL_PATH)

print("預測中...")
probs = model.predict_proba(X)[:, 1]
preds = (probs > THRESHOLD).astype(int)

np.save(OUT_PATH_PROBA, probs)
np.save(OUT_PATH_BIN, preds)

print(f"已儲存：{OUT_PATH_PROBA}, shape = {probs.shape}")
print(f"已儲存：{OUT_PATH_BIN}, shape = {preds.shape}")
print(f"採用最佳 threshold = {THRESHOLD}")
