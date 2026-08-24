from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DataPipeline.config import ABT_PATH, CLEAN_DATA_PATH, RAW_DATA_PATH
from MLOps import storage

DESCRIPTIONS = {
    "SK_ID_CURR": "Identificador da solicitação/cliente.",
    "TARGET": "Alvo supervisionado: 1 = inadimplente; 0 = adimplente.",
    "CREDIT_INCOME_RATIO": "Valor do crédito dividido pela renda total.",
    "ANNUITY_INCOME_RATIO": "Anuidade/parcela dividida pela renda total.",
    "ANNUITY_CREDIT_RATIO": "Anuidade/parcela dividida pelo valor do crédito.",
    "BUREAU_CREDIT_COUNT": "Quantidade de créditos externos encontrados no bureau.",
    "BUREAU_AMT_DEBT_TOTAL": "Soma da dívida aberta registrada no bureau.",
    "BUREAU_DAY_OVERDUE_MAX": "Maior atraso em dias observado no bureau.",
    "BUREAU_DEBT_CREDIT_RATIO": "Dívida total dividida pelo crédito total registrado no bureau.",
    "PREV_APP_COUNT": "Quantidade de solicitações anteriores.",
    "PREV_APPROVAL_RATE": "Proporção de solicitações anteriores aprovadas.",
    "PREV_REFUSED_RATE": "Proporção de solicitações anteriores recusadas.",
}


def table(df):
    lines = [
        "| Coluna | Tipo | % Nulos | Únicos | Descrição |",
        "|---|---:|---:|---:|---|",
    ]
    for col in df.columns:
        if col.startswith("BUREAU_"):
            default_desc = "Feature agregada do histórico de crédito externo."
        elif col.startswith("PREV_"):
            default_desc = "Feature agregada das solicitações anteriores."
        else:
            default_desc = "Feature original do conjunto Home Credit."
        description = DESCRIPTIONS.get(col, default_desc)
        lines.append(
            f"| `{col}` | {df[col].dtype} | {df[col].isna().mean():.1%} | "
            f"{df[col].nunique(dropna=False):,} | {description} |"
        )
    return "\n".join(lines)


def run():
    outputs = [
        (RAW_DATA_PATH, "dicionario_raw_data.md", "raw_data.csv"),
        (CLEAN_DATA_PATH, "dicionario_clean_data.md", "clean_data.csv"),
        (ABT_PATH, "dicionario_abt.md", "abt.csv"),
    ]
    for key, filename, title in outputs:
        if not storage.exists(key):
            print(f"Ignorado: {key} ainda não existe")
            continue
        df = storage.read_csv(key, low_memory=False)
        text = f"# Dicionário — {title}\n\nDimensão: {len(df):,} linhas × {df.shape[1]} colunas.\n\n" + table(df) + "\n"
        (ROOT / "DataPipeline" / filename).write_text(text, encoding="utf-8")
        print(f"Gerado: DataPipeline/{filename}")


if __name__ == "__main__":
    run()
