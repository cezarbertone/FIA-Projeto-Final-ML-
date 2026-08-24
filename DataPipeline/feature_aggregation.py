"""Agregação de bureau e previous_application no nível ``SK_ID_CURR``.

As regras de agregação ficam em ``pipeline_functions.py``. Este script coordena I/O,
persistência e relatório da etapa.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DataPipeline import config
from DataPipeline import pipeline_functions as pf
from MLOps import storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)


def run() -> dict:
    """Agrega as duas fontes históricas para uma linha por cliente."""
    bureau_source = storage.read_csv(config.CLEAN_BUREAU_PATH, low_memory=False)
    previous_source = storage.read_csv(config.CLEAN_PREVIOUS_PATH, low_memory=False)

    bureau = pf.aggregate_bureau_dataframe(bureau_source)
    previous = pf.aggregate_previous_dataframe(previous_source)

    storage.write_csv(bureau, config.FEATURES_BUREAU_PATH)
    storage.write_csv(previous, config.FEATURES_PREVIOUS_PATH)

    LOGGER.info("Bureau agregado: %s clientes | %s features", len(bureau), bureau.shape[1] - 1)
    LOGGER.info("Previous agregado: %s clientes | %s features", len(previous), previous.shape[1] - 1)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bureau_shape": list(bureau.shape),
        "previous_application_shape": list(previous.shape),
        "bureau_features": int(max(0, bureau.shape[1] - 1)),
        "previous_application_features": int(max(0, previous.shape[1] - 1)),
    }
    storage.write_json(report, config.REPORT_PATHS["feature_aggregation"])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    run()
