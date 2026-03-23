import numpy as np
import pandas as pd


CATEGORICAL_FEATURES = [
    'SEX', 'INSR_TYPE', 'TYPE_VEHICLE', 'MAKE', 'USAGE', 'EFFECTIVE_YR',
    'begin_month', 'begin_weekday'
]

NUMERIC_FEATURES = [
    'INSURED_VALUE', 'PREMIUM', 'PROD_YEAR', 'SEATS_NUM', 'CARRYING_CAPACITY',
    'CCM_TON', 'policy_days', 'vehicle_age', 'premium_to_value'
]

DROP_COLUMNS = ['CLAIM_PAID', 'OBJECT_ID', 'INSR_BEGIN', 'INSR_END', 'target']

EXPECTED_RAW_COLUMNS = [
    'SEX', 'INSR_BEGIN', 'INSR_END', 'EFFECTIVE_YR', 'INSR_TYPE', 'INSURED_VALUE',
    'PREMIUM', 'OBJECT_ID', 'PROD_YEAR', 'SEATS_NUM', 'CARRYING_CAPACITY',
    'TYPE_VEHICLE', 'CCM_TON', 'MAKE', 'USAGE', 'CLAIM_PAID'
]

DATE_FMT = '%d-%b-%y'


def parse_mixed_dates(series: pd.Series) -> pd.Series:
    first_try = pd.to_datetime(series, format=DATE_FMT, errors='coerce')
    second_try = pd.to_datetime(series, errors='coerce')
    return first_try.fillna(second_try)


def add_missing_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for col in EXPECTED_RAW_COLUMNS:
        if col not in result.columns:
            result[col] = np.nan
    return result


def prepare_base_features(df: pd.DataFrame) -> pd.DataFrame:
    work = add_missing_raw_columns(df)
    work['INSR_BEGIN'] = parse_mixed_dates(work['INSR_BEGIN'])
    work['INSR_END'] = parse_mixed_dates(work['INSR_END'])
    work['CLAIM_PAID'] = pd.to_numeric(work['CLAIM_PAID'], errors='coerce')
    work['target'] = (work['CLAIM_PAID'].fillna(0) > 0).astype(int)
    work["EFFECTIVE_YR"] = work["EFFECTIVE_YR"].astype("string")
    work['policy_days'] = (work['INSR_END'] - work['INSR_BEGIN']).dt.days
    work['begin_month'] = work['INSR_BEGIN'].dt.month
    work['begin_weekday'] = work['INSR_BEGIN'].dt.weekday
    work['vehicle_age'] = work['INSR_BEGIN'].dt.year - work['PROD_YEAR']
    work['premium_to_value'] = work['PREMIUM'] / work['INSURED_VALUE'].replace(0, np.nan)
    return work


def split_xy(df: pd.DataFrame):
    X = df.drop(columns=DROP_COLUMNS, errors='ignore')
    y = df['target']
    return X, y
