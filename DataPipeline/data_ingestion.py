"""Ingestão dos três CSVs da Landing para a área de dados do projeto.

A Landing é preenchida manualmente. Esta etapa somente promove os arquivos recebidos;
não aplica limpeza, agregação ou regra de modelagem.
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
    """Promove as três fontes da Landing e registra um relatório de ingestão."""
    storage.ensure_bucket()
    items = {}
    for logical_name, (destination, row_limit) in config.INGESTION_SOURCES.items():
        source = config.find_landing_file(logical_name)
        items[logical_name] = pf.promote_landing_file(source, destination, row_limit)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "landing_dir": str(config.LANDING_DIR),
        "storage_backend": storage.STORAGE_BACKEND,
        "files": items,
    }
    storage.write_json(report, config.REPORT_PATHS["ingestion"])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    run()
