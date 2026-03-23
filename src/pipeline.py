from pathlib import Path
import pandas as pd
import joblib
from .common import ensure_dirs, project_path
from .data_collection import calculate_batch_meta, ingest_next_batch
from .data_quality import build_association_rules, clean_batch, evaluate_quality
from .features import parse_mixed_dates, prepare_base_features, split_xy
from .modeling import train_and_select_model
from .reporting import build_summary_report


def normalize_df(df):
    # даты к одному формату
    out = df.copy()
    for col in ['INSR_BEGIN', 'INSR_END']:
        if col in out.columns:
            out[col] = parse_mixed_dates(out[col]).dt.strftime('%Y-%m-%d')
    return out

def append_batch(clean_df, cfg: dict):
    # новый очищенный батч добавляется в общее накопленное хранилище
    clean_master_path = project_path(cfg['paths']['clean_master'])
    clean_master_path.parent.mkdir(parents=True, exist_ok=True)

    clean_df = normalize_df(clean_df)

    if clean_master_path.exists():
        old = pd.read_csv(clean_master_path, dtype={"EFFECTIVE_YR": "string"}, low_memory=False)
        old = normalize_df(old)
        full = pd.concat([old, clean_df], ignore_index=True)
        full = normalize_df(full)
        full.to_csv(clean_master_path, index=False)
    else:
        clean_df.to_csv(clean_master_path, index=False)

    return clean_master_path

def run_update(cfg: dict, logger) -> bool:
    # батч -> quality -> чистка -> в хранилище -> правила -> обучение
    ensure_dirs(cfg)
    batch_id, batch_df, raw_path = ingest_next_batch(cfg)
    logger.info(f'Получен батч: {batch_id} ({raw_path.name})')

    meta = calculate_batch_meta(batch_id, batch_df)
    meta_path = Path(project_path(cfg['paths']['batch_meta']))
    df_row = pd.DataFrame([meta])
    if meta_path.exists():
        old = pd.read_csv(meta_path)
        df_row = pd.concat([old, df_row], ignore_index=True)
    df_row.to_csv(meta_path, index=False)

    quality = evaluate_quality(batch_id, batch_df, cfg)
    logger.info(f"Data quality: duplicates={quality['duplicates']}, missing_rows={quality['rows_with_any_missing']}")

    # очистка батча и сохранение его как отдельный файл
    clean_df = clean_batch(batch_df, cfg)
    clean_path = project_path(cfg['paths']['clean_storage_dir']) / f'clean_{raw_path.name}'
    clean_df.to_csv(clean_path, index=False)
    logger.info(f'Очищенный батч сохранен: {clean_path.name}')
    clean_master_path = append_batch(clean_df, cfg)
    cumulative_df = pd.read_csv(clean_master_path, dtype={"EFFECTIVE_YR": "string"}, low_memory=False)
    cumulative_df['INSR_BEGIN'] = parse_mixed_dates(cumulative_df['INSR_BEGIN'])
    cumulative_df['INSR_END'] = parse_mixed_dates(cumulative_df['INSR_END'])
    
    # правила на накопленных данных
    rules_path = build_association_rules(cumulative_df, cfg, batch_id)
    logger.info(f'Ассоциативные правила сохранены: {rules_path.name}')

    best_model = train_and_select_model(cumulative_df, cfg, batch_id)
    logger.info(
        f"Лучшая модель: {best_model['model_name']} | "
        f"train={best_model['train_rows']} | "
        f"valid={best_model['valid_rows']} | "
        f"valid_pos_rate={best_model['valid_pos_rate']:.6f} | "
        f"PR-AUC={best_model['pr_auc']:.8f} | "
        f"ROC-AUC={best_model['roc_auc']:.8f} | "
        f"F1={best_model['f1']:.8f}"
    )
    return True



def run_inference(cfg: dict, logger, file_path: str) -> str:
    # Для inference только последняя сохраненная лучшая модель
    latest_model_file = project_path(cfg['paths']['latest_model'])
    if not latest_model_file.exists():
        raise RuntimeError('Сначала нужно выполнить update, чтобы появилась модель.')

    model_path = latest_model_file.read_text(encoding='utf-8').strip()
    model = joblib.load(model_path)

    raw = pd.read_csv(file_path, dtype={"EFFECTIVE_YR": "string"}, low_memory=False)
    prepared = prepare_base_features(raw)
    X, _ = split_xy(prepared)

    raw['predict_proba'] = model.predict_proba(X)[:, 1]
    raw['predict'] = (raw['predict_proba'] >= 0.5).astype(int)

    out_path = project_path(cfg['paths']['predictions_dir']) / f"{Path(file_path).stem}_predictions.csv"
    raw.to_csv(out_path, index=False)
    logger.info(f'Предсказания сохранены: {out_path}')
    return str(out_path)



def run_summary(cfg: dict, logger) -> str:
    # отчет
    report_path = build_summary_report(cfg)
    logger.info(f'Отчет сохранен: {report_path}')
    return str(report_path)
