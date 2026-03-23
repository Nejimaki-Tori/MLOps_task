from itertools import combinations
import pandas as pd
from .common import project_path


def evaluate_quality(batch_id: str, df, cfg: dict) -> dict:
    # Функция для оценки качества одного батча
    summary = {
        'batch_id': batch_id,
        'rows': int(len(df)),
        'duplicates': int(df.duplicated().sum()),
        'rows_with_any_missing': int(df.isna().any(axis=1).sum()),
        'missing_share_total': float(df.isna().mean().mean()),
        'invalid_date_order': int((df['INSR_END'] < df['INSR_BEGIN']).fillna(False).sum()),
        'negative_premium': int((df['PREMIUM'] < 0).fillna(False).sum()),
        'negative_insured_value': int((df['INSURED_VALUE'] < 0).fillna(False).sum())
    }

    summary_path = project_path(cfg['paths']['data_quality'])
    
    # сохранение информации на диск
    df_row = pd.DataFrame([summary])
    if summary_path.exists():
        old = pd.read_csv(summary_path)
        df_row = pd.concat([old, df_row], ignore_index=True)
    df_row.to_csv(summary_path, index=False)

    # доля пропусков по каждому столбцу
    missing_by_col = df.isna().mean().reset_index()
    missing_by_col.columns = ['column', 'missing_share']
    missing_by_col['batch_id'] = batch_id
    missing_path = project_path(cfg['paths']['metrics_dir']) / f'missing_by_column_{batch_id}.csv'
    missing_by_col.to_csv(missing_path, index=False)
    return summary


def clean_batch(df, cfg: dict):
    # очистка батча
    cleaned = df.copy()
    if cfg['quality']['drop_duplicates']:
        cleaned = cleaned.drop_duplicates()

    # дата окончания не должна быть раньше даты начала
    cleaned = cleaned[~(cleaned['INSR_END'] < cleaned['INSR_BEGIN']).fillna(False)]
    
    # не должны быть отрицательными
    cleaned = cleaned[~(cleaned['PREMIUM'] < 0).fillna(False)]
    cleaned = cleaned[~(cleaned['INSURED_VALUE'] < 0).fillna(False)]
    return cleaned.reset_index(drop=True)


def apriori_rules(binary_df: pd.DataFrame, min_support: float, min_confidence: float) -> pd.DataFrame:
    # ассоциативные правила между парами бинарных признаков

    rules = []
    support_one = {}
    for col in binary_df.columns:
        support_one[col] = binary_df[col].mean()

    # перебираем все пары
    for left, right in combinations(binary_df.columns, 2):
        both_true = (binary_df[left] & binary_df[right]).mean()

        # не обрабатываются редкие пары
        if both_true < min_support:
            continue


        if support_one[left] > 0:
            conf_left_to_right = both_true / support_one[left]
        else:
            conf_left_to_right = 0
            
        if support_one[right] > 0:
            conf_right_to_left = both_true / support_one[right]
        else:
            conf_right_to_left = 0

        if support_one[right] > 0:
            lift_left_to_right = conf_left_to_right / support_one[right]
        else:
            lift_left_to_right = 0

        if support_one[left] > 0:
            lift_right_to_left = conf_right_to_left / support_one[left]
        else:
            lift_right_to_left = 0

        if conf_left_to_right >= min_confidence:
            rules.append({
                "antecedents": left,
                "consequents": right,
                "support": both_true,
                "confidence": conf_left_to_right,
                "lift": lift_left_to_right
            })
        if conf_right_to_left >= min_confidence:
            rules.append({
                "antecedents": right,
                "consequents": left,
                "support": both_true,
                "confidence": conf_right_to_left,
                "lift": lift_right_to_left
            })

    return pd.DataFrame(rules)


def build_association_rules(df, cfg: dict, batch_id: str):
    # приведение к необходимому виду (true/false) для правил ассоциаций и их построение
    work = df.copy()
    work['vehicle_age'] = work['INSR_BEGIN'].dt.year - work['PROD_YEAR']
    work['policy_days'] = (work['INSR_END'] - work['INSR_BEGIN']).dt.days

    binary = pd.DataFrame({
        'claim_flag': (work['CLAIM_PAID'].fillna(0) > 0),
        'premium_high': work['PREMIUM'] >= work['PREMIUM'].quantile(0.75),
        'insured_value_high': work['INSURED_VALUE'] >= work['INSURED_VALUE'].quantile(0.75),
        'vehicle_old': work['vehicle_age'] >= work['vehicle_age'].median(),
        'policy_long': work['policy_days'] >= work['policy_days'].median(),
        'pickup_vehicle': work['TYPE_VEHICLE'].fillna('').eq('Pick-up'),
        'own_goods_use': work['USAGE'].fillna('').eq('Own Goods')
    }).astype(bool)

    rules = apriori_rules(
        binary,
        min_support=cfg['apriori']['min_support'],
        min_confidence=cfg['apriori']['min_confidence']
    )

    out_path = project_path(cfg['paths']['reports_dir']) / f'association_rules_{batch_id}.csv'
    if rules.empty:
        pd.DataFrame(columns=[
        'antecedents', 'consequents', 'support', 'confidence', 'lift'
        ]).to_csv(out_path, index=False)
        return out_path

    rules = rules.sort_values(['lift', 'confidence'], ascending=False).head(cfg['apriori']['top_k_rules'])
    rules.to_csv(out_path, index=False)
    return out_path
