# train_stacking_meta_catb.py
# 使用 CatBoost 作為 meta-model，融合 p_catb + p_xgb 預測

import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# === 讀取資料 ===
p_catb = np.load("stock_predict_5/p_catb_training.npy")
p_xgb = np.load("stock_predict_5/p_xgb_training.npy")
y = np.load("stock_predict_5/y.npy")

X_meta = np.vstack([p_catb, p_xgb]).T

# === 分割訓練與驗證集 ===
X_train, X_val, y_train, y_val = train_test_split(X_meta, y, test_size=0.2, stratify=y, random_state=42)
train_pool = Pool(X_train, y_train)
val_pool = Pool(X_val, y_val)

# === 訓練 CatBoost meta model ===
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    eval_metric='Logloss',
    loss_function='Logloss',
    random_seed=42,
    verbose=50,
    early_stopping_rounds=30
)

model.fit(train_pool, eval_set=val_pool)
model.save_model("stock_predict_5/meta_model_catb.cbm")

# === 預測與 threshold 掃描 ===
p_val = model.predict_proba(X_val)[:, 1]
np.save("stock_predict_5/p_stack_train_catb.npy", p_val)

best_f1 = 0
best_t = 0
for t in np.linspace(0.01, 0.99, 100):
    preds = (p_val >= t).astype(int)
    f1 = f1_score(y_val, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_t = t

y_pred = (p_val >= best_t).astype(int)

# === 統計輸出 ===
print(f"\nCatBoost Meta F1-score = {best_f1:.4f} @ threshold = {best_t:.4f}")
print("Precision =", precision_score(y_val, y_pred))
print("Recall    =", recall_score(y_val, y_pred))

# === 畫圖 ===
plt.figure(figsize=(10, 6))
plt.hist(p_val, bins=50, color='orange', edgecolor='black', log=True)
plt.axvline(best_t, color='red', linestyle='--', label=f"Best Threshold = {best_t:.4f}")
plt.title("Meta CatBoost Prediction Distribution")
plt.xlabel("Predicted Probability")
plt.ylabel("Log Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
