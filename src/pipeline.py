"""建立資料預處理Pipeline（ColumnTransformer）。"""
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder

from src.features import GroupSleepDiffTransformer, engineered_features

ORDINAL_COLS = ['stress_level', 'sleep_quality', 'physical_activity_level', 'smoking_alcohol']
ORDINAL_CATEGORIES = [
    ['missing', 'low', 'medium', 'high'],           # stress_level
    ['missing', 'poor', 'average', 'good'],         # sleep_quality
    ['missing', 'sedentary', 'moderate', 'active'], # physical_activity_level
    ['missing', 'no', 'occasional', 'yes'],         # smoking_alcohol
]
NOMINAL_COLS = ['diet_type', 'gender', 'stress_activity_interaction']


def build_preprocessor():
    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OrdinalEncoder(
            categories=ORDINAL_CATEGORIES,
            handle_unknown='use_encoded_value',
            unknown_value=-1,
        )),
    ])

    nominal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    column_transformer = ColumnTransformer(transformers=[
        ('num', 'passthrough', make_column_selector(dtype_include=['number'])),
        ('ordinal_col', ordinal_transformer, ORDINAL_COLS),
        ('nominal_col', nominal_transformer, NOMINAL_COLS),
    ])

    preprocessor = Pipeline(steps=[
        ('feature_engineer', FunctionTransformer(engineered_features)),
        ('sleep_diff', GroupSleepDiffTransformer()),
        ('column_transformer', column_transformer),
    ])
    preprocessor.set_output(transform='pandas')
    return preprocessor
