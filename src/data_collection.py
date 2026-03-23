import json
import shutil
from pathlib import Path
import pandas as pd
from .common import project_path


def read_source_csv(source_path):
    # Чтение датасета и преобразование к нужному формату
    df = pd.read_csv(
        source_path,
        dtype={"EFFECTIVE_YR": "string"},
        low_memory=False
    )

    df["INSR_BEGIN"] = pd.to_datetime(df["INSR_BEGIN"], format="%d-%b-%y", errors="coerce")
    df["INSR_END"] = pd.to_datetime(df["INSR_END"], format="%d-%b-%y", errors="coerce")

    return df.sort_values("INSR_BEGIN").reset_index(drop=True)


def prepare_batches(cfg: dict):
    # Разбиение датасета на батчи
    source_path = project_path(cfg['paths']['source_data'])
    batches_dir = project_path(cfg['paths']['batches_dir'])

    existing = sorted(batches_dir.glob('batch_*.csv'))
    if existing: # вернуть, если уже есть
        return existing

    df = read_source_csv(source_path)
    df['batch_id'] = df['INSR_BEGIN'].dt.to_period(cfg['data']['batch_freq']).astype(str) # разбиваем данные на батчи

    created_files = []
    for i, (batch_id, batch_df) in enumerate(df.groupby('batch_id'), start=1):
        # Сохранение батчей в файлы
        file_name = f'batch_{i:03d}_{batch_id}.csv'
        out_path = batches_dir / file_name
        batch_df.drop(columns='batch_id').to_csv(out_path, index=False)
        created_files.append(out_path)
        
    return created_files

def ingest_next_batch(cfg: dict):
    # Получение новой порции данных
    batch_files = prepare_batches(cfg)
    state_path = project_path(cfg['paths']['state_file'])
    raw_storage_dir = project_path(cfg['paths']['raw_storage_dir'])
    state_path.parent.mkdir(parents=True, exist_ok=True)

    if not state_path.exists():
        state = {'next_batch_index': 0}
    else:
        with open(state_path, 'r', encoding='utf-8') as f:
            state = json.load(f)
            
    idx = state['next_batch_index']
    if idx >= len(batch_files):
        raise RuntimeError('Новых батчей больше нет.')

    batch_path = batch_files[idx]
    raw_copy_path = raw_storage_dir / batch_path.name
    shutil.copy2(batch_path, raw_copy_path) # сохранение файла нового батча в систему

    batch_df = pd.read_csv(raw_copy_path)
    batch_df["INSR_BEGIN"] = pd.to_datetime(batch_df["INSR_BEGIN"], errors="coerce")
    batch_df["INSR_END"] = pd.to_datetime(batch_df["INSR_END"], errors="coerce")

    
    state['next_batch_index'] = idx + 1
    with open(state_path, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

    batch_id = raw_copy_path.stem.replace('batch_', '')
    return batch_id, batch_df, raw_copy_path


def calculate_batch_meta(batch_id: str, df) -> dict:
    target = (df['CLAIM_PAID'].fillna(0) > 0).astype(int)
    return {
        'batch_id': batch_id,
        'rows': int(len(df)),
        'columns': int(df.shape[1]),
        'date_min': str(df['INSR_BEGIN'].min().date()) if df['INSR_BEGIN'].notna().any() else None,
        'date_max': str(df['INSR_BEGIN'].max().date()) if df['INSR_BEGIN'].notna().any() else None,
        'target_rate': float(target.mean()),
        'missing_share_total': float(df.isna().mean().mean())
    }
