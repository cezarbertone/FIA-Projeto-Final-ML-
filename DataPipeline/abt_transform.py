"""Geração de ``Dados/abt.csv`` com uma linha por ``SK_ID_CURR``.

As regras de feature engineering, joins, preenchimento do histórico e seleção por
missing rate ficam centralizadas em ``pipeline_functions.py``.
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


def _read_optional_features(path: str):
    """Lê uma tabela agregada quando ela já foi produzida."""
    return storage.read_csv(path, low_memory=False) if storage.exists(path) else None


def run() -> dict:
    """Constrói e persiste a ABT usada pelo treinamento."""
    application = pf.load_clean_data()
    bureau_features = _read_optional_features(config.FEATURES_BUREAU_PATH)
    previous_features = _read_optional_features(config.FEATURES_PREVIOUS_PATH)

    abt, dropped = pf.build_abt_dataframe(application, bureau_features, previous_features)
    storage.write_csv(abt, config.ABT_PATH)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(abt)),
        "columns": int(abt.shape[1]),
        "model_input_columns_before_encoding": int(abt.shape[1] - 2),
        "default_rate": float(abt[config.TARGET_COL].mean()),
        "bureau_feature_count": int(sum(c.startswith("BUREAU_") for c in abt.columns)),
        "previous_feature_count": int(sum(c.startswith("PREV_") for c in abt.columns)),
        "dropped_high_null_columns": dropped,
    }
    storage.write_json(report, config.REPORT_PATHS["abt"])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    run()
