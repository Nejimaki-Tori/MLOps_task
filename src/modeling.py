from datetime import datetime
from pathlib import Path
import numpy as np
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .common import project_path
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, prepare_base_features, split_xy


def _stream_holdout(df, batch_id: str):
    work = df.sort_values('INSR_BEGIN').reset_index(drop=True).copy()
    work['stream_batch'] = work['INSR_BEGIN'].dt.to_period('M').astype(str)

    # 013_2017-07 -> 2017-07
    current_batch = batch_id.split('_', 1)[1]

    train_df = work[work['stream_batch'] < current_batch].copy()

    # Текущий батч используется как валидация
    valid_df = work[work['stream_batch'] == current_batch].copy()

    # запасной вариант 80/20 по времени
    if len(train_df) == 0 or len(valid_df) == 0:
        split_idx = int(len(work) * 0.8)
        split_idx = max(1, min(split_idx, len(work) - 1))
        train_df = work.iloc[:split_idx].copy()
        valid_df = work.iloc[split_idx:].copy()

    return train_df, valid_df


def _limit_train_rows(df, max_rows: int):
    # ограничение объем train если указано в конфиге
    if max_rows is None or len(df) <= max_rows:
        return df
    return df.sort_values('INSR_BEGIN').tail(max_rows).copy()


def build_tree_model(random_state: int):
    # ансамбль деревьев
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median'))
        ]), NUMERIC_FEATURES),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ]), CATEGORICAL_FEATURES)
    ])

    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=20,
            class_weight='balanced_subsample',
            random_state=random_state,
            n_jobs=-1
        ))
    ])


def build_mlp_model(random_state: int) -> Pipeline:
    # для нейросети числовые признаки нормализуются, а категории превратить в one-hot, чтобы ей было проще с ними работать
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), NUMERIC_FEATURES),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', min_frequency=50))
        ]), CATEGORICAL_FEATURES)
    ])

    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', MLPClassifier(
            hidden_layer_sizes=(32, 16),
            max_iter=100,
            early_stopping=True,
            random_state=random_state
        ))
    ])

def _safe_roc_auc(y_true: pd.Series, y_proba: pd.Series) -> float:
    # ROC-AUC нельзя посчитать, если в валидации только один класс
    if y_true.nunique() < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_proba))
    
def find_best_threshold(y_true, y_proba):
    # поиск лучшего порога для отсечения
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in np.arange(0.05, 0.51, 0.05):
        pred = (y_proba >= threshold).astype(int)
        score = f1_score(y_true, pred, zero_division=0)

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return best_threshold, best_f1

def train_and_select_model(df: pd.DataFrame, cfg: dict, batch_id: str) -> dict:
    # подготовка признаков и удаление строк без даты начала полиса
    work = prepare_base_features(df)
    work = work.dropna(subset=['INSR_BEGIN']).copy()

    train_df, valid_df = _stream_holdout(work, batch_id)
    train_df = _limit_train_rows(train_df, cfg['training']['train_max_rows'])

    X_train, y_train = split_xy(train_df)
    X_valid, y_valid = split_xy(valid_df)

    run_version = datetime.now().strftime('%Y%m%d_%H%M%S')
    models_dir = project_path(cfg['paths']['models_dir'])
    registry_path = project_path(cfg['paths']['model_registry'])

    models = {
        'random_forest': build_tree_model(cfg['training']['random_state']),
        'mlp': build_mlp_model(cfg['training']['random_state'])
    }

    registry_rows = []
    best_row = None

    for model_name, pipeline in models.items():
        pipeline.fit(X_train, y_train)

        valid_proba = pipeline.predict_proba(X_valid)[:, 1]
        best_threshold, best_f1 = find_best_threshold(y_valid, valid_proba)
        valid_pred = (valid_proba >= best_threshold).astype(int)

        row = {
            'run_version': run_version,
            'batch_id': batch_id,
            'model_name': model_name,
            'train_rows': int(len(train_df)),
            'valid_rows': int(len(valid_df)),
            'valid_pos_rate': float(y_valid.mean()),
            'roc_auc': _safe_roc_auc(y_valid, valid_proba),
            'pr_auc': float(average_precision_score(y_valid, valid_proba)),
            'f1': float(f1_score(y_valid, valid_pred, zero_division=0)),
            'model_path': str((models_dir / f'{run_version}_{model_name}.joblib').resolve())
        }

        joblib.dump(pipeline, row['model_path'])
        registry_rows.append(row)

        if best_row is None or row[cfg['validation']['main_metric']] > best_row[cfg['validation']['main_metric']]:
            best_row = row

    new_df = pd.DataFrame(registry_rows)
    if registry_path.exists():
        old = pd.read_csv(registry_path)
        new_df = pd.concat([old, new_df], ignore_index=True)
    new_df.to_csv(registry_path, index=False)

    latest_model_path = project_path(cfg['paths']['latest_model'])
    latest_model_path.write_text(best_row['model_path'], encoding='utf-8')

    return best_row