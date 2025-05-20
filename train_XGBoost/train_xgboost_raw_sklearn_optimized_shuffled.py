
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import joblib

# ==== 超參數設定（最佳實務建議）====
LEARNING_RATE = 0.01
MAX_DEPTH = 8
MIN_CHILD_WEIGHT = 1
GAMMA = 0.01
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8
N_ESTIMATORS = 400
EARLY_STOPPING_ROUNDS = 20
TREE_METHOD = "hist"
GROW_POLICY = "lossguide"
SCALE_POS_WEIGHT = 9.5  
N_JOBS = 8
RANDOM_SEED = 42
TEST_SIZE = 0.1

def train_xgboost(X_path="training_dataset_procced/X_full.npy", y_path="training_dataset_procced/y.npy"):
    print("📦 載入資料...")
    X = np.load(X_path)
    y = np.load(y_path)

    print("🔀 打散資料中（shuffle）...")
    X, y = shuffle(X, y, random_state=RANDOM_SEED)

    print("📊 分割訓練 / 驗證集...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED)

    print("🧠 開始訓練 XGBClassifier...")
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        min_child_weight=MIN_CHILD_WEIGHT,
        gamma=GAMMA,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        n_estimators=N_ESTIMATORS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        scale_pos_weight= sum(y == 0) / sum(y == 1),
        tree_method=TREE_METHOD,
        grow_policy=GROW_POLICY,
        n_jobs=N_JOBS,
        verbosity=1,
        random_state=RANDOM_SEED,
        use_label_encoder=False
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    joblib.dump(model, "xgb_raw_model.pkl")
    print("已儲存為：xgb_raw_model.pkl")

    print("預測訓練資料...")
    probs = model.predict_proba(X)[:, 1]
    np.save("p_xgb.npy", probs)
    print(f"已儲存：p_xgb.npy, shape = {probs.shape}")

    # 計算最佳 F1-score
    best_f1, best_t = 0, 0
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (probs > t).astype(int)
        f1 = f1_score(y, preds)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"最佳 threshold = {best_t:.2f}, F1-score = {best_f1:.4f}")

if __name__ == "__main__":
    train_xgboost()
