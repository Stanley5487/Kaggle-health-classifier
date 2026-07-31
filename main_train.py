"""訓練：讀取 data/train.csv，訓練LightGBM並存檔至 models/。
**註:原本是是永XGB，但發現LGB更準，故在此只放LGB
超參數（config/best_lgb_params.json）為先前 Optuna 5-fold CV 搜尋（20 trials，
以 multi_logloss 為目標）所得的最佳組合，對應 public LB 0.94960 的最佳提交版本。
訓練時對三個類別使用 sqrt(balanced sample_weight) 加權，緩解類別不平衡。

註:原本是是永XGB，但發現LGB更準，故在此只放LGB，XGB的訓練過程可參考notebook中的檔案
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from src.pipeline import build_preprocessor

DATA_DIR = Path('data')
CONFIG_PATH = Path('config/best_lgb_params.json')
MODEL_PATH = Path('models/health_lgb_mod3.pkl')

LABEL_ORDER = ['at-risk', 'fit', 'unhealthy']


def main():
    df = pd.read_csv(DATA_DIR / 'train.csv')
    X = df.loc[:, [c for c in df.columns if c not in ('id', 'health_condition')]]

    label_encoder = LabelEncoder().fit(LABEL_ORDER)
    y = label_encoder.transform(df['health_condition'])

    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        lgb_params = json.load(f)
    lgb_params.setdefault('random_state', 42)

    model = Pipeline(steps=[
        ('preprocessor', build_preprocessor()),
        ('lgb', LGBMClassifier(**lgb_params)),
    ])

    sample_weight = np.sqrt(compute_sample_weight('balanced', y))
    model.fit(X, y, lgb__sample_weight=sample_weight)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f'模型已儲存至 {MODEL_PATH}')
    print('類別對應:', dict(enumerate(label_encoder.classes_)))


if __name__ == '__main__':
    main()
