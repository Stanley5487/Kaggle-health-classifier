"""特徵工程：缺失值旗標、壓力x活動交互特徵，以及群組相對偏離度轉換器。"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def engineered_features(df):
    """加入缺失值計數與 stress x activity 交互特徵。

    在特徵重要性分析中發現，`stress_level` / `physical_activity_level` 是否為
    缺失值本身就帶有強烈的風險行為訊號（MNAR，非隨機缺失），重要性甚至超越
    BMI、心率等生理指標，因此在此明確地將缺失模式與交互情境編碼出來。
    """
    df = df.copy()

    missing_sleep = df['sleep_duration'].isna().astype(int)
    missing_stress = df['stress_level'].isna().astype(int)
    missing_activity = df['physical_activity_level'].isna().astype(int)
    df['missing_key_counts'] = missing_sleep + missing_stress + missing_activity

    has_both = df['stress_level'].notna() & df['physical_activity_level'].notna()
    df['stress_activity_interaction'] = np.where(
        has_both,
        df['stress_level'].astype(str) + '_' + df['physical_activity_level'].astype(str),
        np.nan,
    )
    return df


class GroupSleepDiffTransformer(BaseEstimator, TransformerMixin):
    """計算數值欄位相對於「壓力 x 活動」情境群體平均值的偏離度。

    單看睡眠時數的絕對值無法反映真實生理負擔：同樣睡 6 小時，在「高壓 + 久坐」
    的情境下可能是嚴重不足，但在「低壓 + 規律運動」下卻可能相對充裕。此轉換器
    以情境群體平均為基準，計算相對偏離度，讓模型看到的是「相對脆弱程度」而非
    單純的絕對數值。

    Ablation 結果：預設只對 `sleep_duration` 做偏離度轉換為最佳設定
    （public LB 0.94960）；額外加入 `bmi` / `step_count` / `exercise_duration`
    三欄位反而讓分數略降至 0.94937，推測為過擬合雜訊，因此不建議擴充。
    """

    def __init__(self, group_col='stress_activity_interaction', target_cols=None):
        self.group_col = group_col
        self.target_cols = target_cols if target_cols else ['sleep_duration']

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.group_means_ = {}
        self.global_means_ = {}
        for col in self.target_cols:
            self.group_means_[col] = X.groupby(self.group_col)[col].mean().to_dict()
            self.global_means_[col] = X[col].mean()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.target_cols:
            group_mean = (
                X[self.group_col]
                .map(self.group_means_[col])
                .fillna(self.global_means_[col])
            )
            X[f'{col}_diff_from_group_mean'] = X[col] - group_mean
        return X

    def get_feature_names_out(self, input_features=None):
        input_features = (
            input_features if input_features is not None else self.feature_names_in_
        )
        new_cols = [f'{col}_diff_from_group_mean' for col in self.target_cols]
        return np.asarray(list(input_features) + new_cols, dtype=object)
