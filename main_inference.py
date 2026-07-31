"""推論:載入模型、讀取 data/test.csv，套用開根號反比先驗權重後輸出submission.csv。

背景：訓練集三個類別（at-risk / fit / unhealthy）分布並不均勻，模型輸出的原始
機率會偏向多數類別。做法是先從 train.csv 精確計算各類別的真實比例（exact priors），
再將 predict_proba 乘上 1/sqrt(prior) 的權重，把預測「推」回接近真實類別比例。
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

DATA_DIR = Path('data')
MODEL_PATH = Path('models/health_lgb_mod3.pkl')
OUTPUT_PATH = Path('submission.csv')

LABEL_MAPPING = {'at-risk': 0, 'fit': 1, 'unhealthy': 2}
REVERSE_LABEL = {v: k for k, v in LABEL_MAPPING.items()}


def main():
    df_train = pd.read_csv(DATA_DIR / 'train.csv')
    df_test = pd.read_csv(DATA_DIR / 'test.csv')
    model = joblib.load(MODEL_PATH)

    X_test = df_test.loc[:, [c for c in df_test.columns if c != 'id']]
    raw_probs = model.predict_proba(X_test)

    # 計算訓練集先驗機率 (exact priors)
    y_train_mapped = df_train['health_condition'].map(LABEL_MAPPING)
    prior_counts = y_train_mapped.value_counts(normalize=True).sort_index().values

    # 開根號反比加權 (sqrt inverse frequency)
    prior_weights = 1.0 / np.sqrt(prior_counts)
    prior_weights = prior_weights / np.mean(prior_weights)

    weighted_probs = raw_probs * prior_weights
    preds = np.argmax(weighted_probs, axis=1)
    labels = [REVERSE_LABEL[p] for p in preds]

    submission = pd.DataFrame({'id': df_test['id'], 'health_condition': labels})
    submission.to_csv(OUTPUT_PATH, index=False)

    print(f'已輸出預測結果至 {OUTPUT_PATH}')
    print('套用的開根號反比權重:', np.round(prior_weights, 4))
    print('預測類別分布比例:')
    print(pd.Series(labels).value_counts(normalize=True))


if __name__ == '__main__':
    main()
