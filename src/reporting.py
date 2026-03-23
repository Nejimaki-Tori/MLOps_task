from pathlib import Path

import pandas as pd

from .common import project_path


def build_summary_report(cfg: dict):
    # пути из конфига в Path
    batch_meta_path = project_path(cfg['paths']['batch_meta'])
    quality_path = project_path(cfg['paths']['data_quality'])
    registry_path = project_path(cfg['paths']['model_registry'])
    reports_dir = project_path(cfg['paths']['reports_dir'])
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Будущий отчет - список строк
    lines = ['# Monitoring summary', '']

    if batch_meta_path.exists():
        batch_meta = pd.read_csv(batch_meta_path)
        lines += ['## Batch meta', '', batch_meta.tail(10).to_markdown(index=False), '']

    if quality_path.exists():
        quality = pd.read_csv(quality_path)
        lines += ['## Data quality', '', quality.tail(10).to_markdown(index=False), '']

    # последние записи и отдельно лучшая модель по PR-AUC
    if registry_path.exists():
        registry = pd.read_csv(registry_path)
        lines += ['## Model registry', '', registry.tail(10).to_markdown(index=False), '']
        best = registry.sort_values('pr_auc', ascending=False).iloc[0]
        lines += [
            '## Best model so far', '',
            f"- model_name: {best['model_name']}",
            f"- pr_auc: {best['pr_auc']:.4f}",
            f"- roc_auc: {best['roc_auc']:.4f}",
            f"- path: {best['model_path']}",
            ''
        ]

    out_path = reports_dir / 'summary_report.md'
    out_path.write_text('\n'.join(lines), encoding='utf-8')
    return out_path
