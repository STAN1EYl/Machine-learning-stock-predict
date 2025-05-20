
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import joblib

# ==== 超參數設定 ====
LEARNING_RATE = 0.03
DEPTH = 9#6x
ITERATIONS = 1000
L2_LEAF_REG = 5#3x
BAGGING_TEMPERATURE = 0.8
RANDOM_STRENGTH = 4
EARLY_STOPPING_ROUNDS = 50
RANDOM_SEED = 42
TEST_SIZE = 0.1
THRESHOLD_STEP = 0.01
THRESHOLD_RANGE = (0.1, 0.9)

def train_catboost(X_path="training_dataset_procced/X_static_catboost.csv", y_path="training_dataset_procced/y.npy"):
    print("載入資料...")
    X = pd.read_csv(X_path)
    y = np.load(y_path)

    print("分割訓練 / 驗證集...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED)

    train_pool = Pool(X_train, label=y_train)
    val_pool = Pool(X_val, label=y_val)
    full_pool = Pool(X, label=y)

    print("開始訓練 CatBoostClassifier...")
    model = CatBoostClassifier(
        iterations=ITERATIONS,
        learning_rate=LEARNING_RATE,
        depth=DEPTH,
        l2_leaf_reg=L2_LEAF_REG,
        bagging_temperature=BAGGING_TEMPERATURE,
        random_strength=RANDOM_STRENGTH,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=RANDOM_SEED,
        verbose=False,
        task_type="CPU",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS
    )

    model.fit(train_pool, eval_set=val_pool, verbose=True)

    model.save_model("catboost_raw_model.cbm")
    print("模型已儲存為：catboost_raw_model.cbm")

    print("預測訓練資料機率中...")
    probs = model.predict_proba(full_pool)[:, 1]
    np.save("p_catboost.npy", probs)
    print(f"已儲存：p_catboost.npy, shape = {probs.shape}")

    # 計算最佳 F1-score 門檻
    best_f1, best_t = 0, 0
    for t in np.arange(*THRESHOLD_RANGE, THRESHOLD_STEP):
        preds = (probs > t).astype(int)
        f1 = f1_score(y, preds)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"最佳 threshold = {best_t:.2f}, F1-score = {best_f1:.4f}")

if __name__ == "__main__":
    train_catboost()
