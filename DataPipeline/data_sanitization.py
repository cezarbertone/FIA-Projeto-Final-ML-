"""Limpeza das três fontes e geração de ``Dados/clean_data.csv``.

As regras de transformação ficam centralizadas em ``pipeline_functions.py``. Este
script apenas lê, aplica as funções reutilizáveis, persiste as saídas e registra o
relatório da etapa.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DataPipeline import config
from DataPipeline import pipeline_functions as pf
from MLOps import storage


def run() -> dict:
    """Limpa application, bureau e previous_application preservando a granularidade."""
    app = pf.sanitize_application_dataframe(storage.read_csv(config.RAW_DATA_PATH, low_memory=False))
    bureau = pf.sanitize_bureau_dataframe(storage.read_csv(config.RAW_BUREAU_PATH, low_memory=False))
    previous = pf.sanitize_previous_dataframe(storage.read_csv(config.RAW_PREVIOUS_PATH, low_memory=False))

    storage.write_csv(app, config.CLEAN_DATA_PATH)
    storage.write_csv(bureau, config.CLEAN_BUREAU_PATH)
    storage.write_csv(previous, config.CLEAN_PREVIOUS_PATH)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "clean_data": {
            "rows": int(len(app)),
            "columns": int(app.shape[1]),
            "default_rate": float(app[config.TARGET_COL].mean()),
        },
        "clean_bureau": {"rows": int(len(bureau)), "columns": int(bureau.shape[1])},
        "clean_previous_application": {"rows": int(len(previous)), "columns": int(previous.shape[1])},
    }
    storage.write_json(report, config.REPORT_PATHS["data_sanitization"])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    run()
