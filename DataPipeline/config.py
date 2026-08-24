from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
LANDING_DIR = PROJECT_ROOT / "Landing"

# Caminhos lógicos. No modo local são arquivos; no MinIO são object keys.
RAW_DATA_PATH = "Dados/raw_data.csv"
CLEAN_DATA_PATH = "Dados/clean_data.csv"
ABT_PATH = "Dados/abt.csv"
PROCESSING_DIR = "Dados/_processing"

RAW_BUREAU_PATH = f"{PROCESSING_DIR}/raw_bureau.csv"
RAW_PREVIOUS_PATH = f"{PROCESSING_DIR}/raw_previous_application.csv"
CLEAN_BUREAU_PATH = f"{PROCESSING_DIR}/clean_bureau.csv"
CLEAN_PREVIOUS_PATH = f"{PROCESSING_DIR}/clean_previous_application.csv"
FEATURES_BUREAU_PATH = f"{PROCESSING_DIR}/features_bureau.csv"
FEATURES_PREVIOUS_PATH = f"{PROCESSING_DIR}/features_previous_application.csv"

MODEL_PATH = "Model/model.pkl"
ABT_ARTIFACTS_PATH = "DataPipeline/abt_artifacts.pkl"
METRICS_PATH = "Model/metrics.json"

RAW_FILES = {
    "application_train": "application_train.csv",
    "bureau": "bureau.csv",
    "previous_application": "previous_application.csv",
}

TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"
DAYS_EMPLOYED_ANOMALY = 365243
NULL_DROP_THRESHOLD = float(os.getenv("NULL_DROP_THRESHOLD", "0.50"))
PROTECTED_NULL_COLUMNS = {"EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"}
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
RISK_THRESHOLD = float(os.getenv("RISK_THRESHOLD", "0.50"))

# Limites opcionais para teste rápido. 0 = arquivo completo.
MAX_ROWS_APPLICATION = int(os.getenv("MAX_ROWS_APPLICATION", "0") or 0)
MAX_ROWS_BUREAU = int(os.getenv("MAX_ROWS_BUREAU", "0") or 0)
MAX_ROWS_PREVIOUS = int(os.getenv("MAX_ROWS_PREVIOUS", "0") or 0)

INGESTION_SOURCES = {
    "application_train": (RAW_DATA_PATH, MAX_ROWS_APPLICATION),
    "bureau": (RAW_BUREAU_PATH, MAX_ROWS_BUREAU),
    "previous_application": (RAW_PREVIOUS_PATH, MAX_ROWS_PREVIOUS),
}

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "home-credit-risk-v3")
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local").lower()

# Listas e caminhos compartilhados entre scripts e notebooks.
PREVIOUS_SENTINEL_COLUMNS = (
    "DAYS_FIRST_DRAWING",
    "DAYS_FIRST_DUE",
    "DAYS_LAST_DUE_1ST_VERSION",
    "DAYS_LAST_DUE",
    "DAYS_TERMINATION",
)

BUREAU_AGG_SOURCE_COLUMNS = (
    "SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "DAYS_CREDIT",
    "CREDIT_DAY_OVERDUE", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT",
    "AMT_CREDIT_SUM_OVERDUE", "CNT_CREDIT_PROLONG",
)

PREVIOUS_AGG_SOURCE_COLUMNS = (
    "SK_ID_PREV", "SK_ID_CURR", "NAME_CONTRACT_STATUS", "AMT_CREDIT",
    "AMT_APPLICATION", "AMT_DOWN_PAYMENT", "DAYS_DECISION", "CNT_PAYMENT",
)

HISTORY_PREFIXES = ("BUREAU_", "PREV_")
RATIO_COLUMNS = (
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "ANNUITY_CREDIT_RATIO",
)
EXT_SOURCE_COLUMNS = ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")
EDA_PROFILE_COLUMNS = (
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "DAYS_BIRTH", "DAYS_EMPLOYED", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
)
EDA_CATEGORICAL_COLUMNS = (
    "NAME_INCOME_TYPE", "ORGANIZATION_TYPE", "NAME_EDUCATION_TYPE",
    "CODE_GENDER", "NAME_FAMILY_STATUS", "NAME_CONTRACT_TYPE",
)
ABT_BUREAU_CHECK_COLUMNS = (
    "BUREAU_CREDIT_COUNT", "BUREAU_ACTIVE_RATIO", "BUREAU_AMT_DEBT_TOTAL",
    "BUREAU_DAY_OVERDUE_MAX", "BUREAU_DEBT_CREDIT_RATIO",
)
ABT_PREVIOUS_CHECK_COLUMNS = (
    "PREV_APP_COUNT", "PREV_APPROVED_COUNT", "PREV_REFUSED_COUNT",
    "PREV_APPROVAL_RATE", "PREV_REFUSED_RATE", "PREV_AMT_CREDIT_MEAN",
)
AGE_BUCKET_BINS = (18, 30, 40, 50, 60, 100)

REPORT_PATHS = {
    "ingestion": "reports/ingestion_report.json",
    "data_sanitization": "reports/data_sanitization_report.json",
    "feature_aggregation": "reports/feature_aggregation_report.json",
    "abt": "reports/abt_report.json",
    "model_comparison": "reports/model_comparison.csv",
    "grid_search": "reports/grid_search_results.csv",
    "holdout_predictions": "reports/holdout_predictions.csv",
    "threshold_analysis": "reports/threshold_analysis.csv",
    "roc_curve": "reports/roc_curve_best_model.csv",
    "feature_importance": "reports/feature_importance.csv",
    "permutation_importance": "reports/permutation_importance.csv",
    "shap_importance": "reports/shap_importance.csv",
    "shap_status": "reports/shap_status.json",
}


def find_landing_file(logical_name: str) -> Path:
    """Localiza o CSV correspondente na Landing, aceitando sufixos no nome do arquivo."""
    filename = RAW_FILES[logical_name]
    base = filename.replace(".csv", "")
    candidates: list[Path] = []
    for pattern in (f"{base}.csv", f"{base}*.csv", f"{base}.CSV", f"{base}*.CSV"):
        candidates.extend(sorted(LANDING_DIR.glob(pattern)))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"{filename} não encontrado em {LANDING_DIR}. "
        "Coloque application_train.csv, bureau.csv e previous_application.csv na pasta Landing."
    )
