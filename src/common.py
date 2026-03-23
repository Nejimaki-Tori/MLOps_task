import json
import logging
from pathlib import Path

# корень проекта (работает для запуска из любой папки)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(relative_path: str) -> Path:
    # Получение абсолютного путя проекта
    return (PROJECT_ROOT / relative_path).resolve()


def load_config() -> dict:
    # Загрузка конфига
    try:
        with open(PROJECT_ROOT / 'config.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        print('No config!')
        return None


def ensure_dirs(cfg: dict) -> None:
    # Создание всех необходимых папок для проекта
    path_keys = [
        'batches_dir', 
        'raw_storage_dir', 
        'clean_storage_dir', 
        'artifacts_dir',
        'models_dir', 
        'metrics_dir', 
        'reports_dir', 
        'predictions_dir',
    ]
    
    for key in path_keys:
        project_path(cfg['paths'][key]).mkdir(parents=True, exist_ok=True)


def get_logger() -> logging.Logger:
    # Логгер
    logger = logging.getLogger('mlops')
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    logger.addHandler(handler)
    return logger
