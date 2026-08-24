"""Predição do score de risco usando dois artefatos explícitos de produção.

Fluxo de inferência:
1. ``DataPipeline/abt_artifacts.pkl`` recria o contrato de entrada da ABT e
   aplica o preprocessing ajustado no treinamento;
2. ``Model/model.pkl`` recebe a matriz já transformada e calcula a PD;
3. o threshold converte a PD em decisão de crédito.

O Streamlit usa este módulo, portanto a aplicação também depende explicitamente
dos dois arquivos.
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DataPipeline import pipeline_functions as pf
from MLOps import storage

_cfg_path = Path(__file__).with_name("config.py")
_spec = importlib.util.spec_from_file_location("model_config", _cfg_path)
config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(config)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)
DECISION_THRESHOLD = config.RISK_THRESHOLD


def load_model(model_path: str | None = None) -> dict:
    """Carrega o estimador treinado e o threshold salvos em ``model.pkl``."""
    path = model_path or config.MODEL_PATH
    if not storage.exists(path):
        raise FileNotFoundError(f"Modelo não encontrado: {path}. Execute o treinamento primeiro.")
    return storage.read_pickle(path)


def load_abt_artifacts(artifacts_path: str | None = None) -> dict:
    """Carrega preprocessing e schema salvos em ``abt_artifacts.pkl``."""
    path = artifacts_path or config.ABT_ARTIFACTS_PATH
    if not storage.exists(path):
        raise FileNotFoundError(
            f"Artefatos da ABT não encontrados: {path}. Execute o treinamento primeiro."
        )
    return storage.read_pickle(path)


def _as_dataframe(records: Union[dict, list, pd.DataFrame]) -> pd.DataFrame:
    """Normaliza dict, lista de dicts ou DataFrame para um DataFrame independente."""
    if isinstance(records, dict):
        return pd.DataFrame([records])
    if isinstance(records, list):
        return pd.DataFrame(records)
    if isinstance(records, pd.DataFrame):
        return records.copy()
    raise TypeError("records deve ser dict, lista de dicts ou pandas.DataFrame")


def prepare_features(df: pd.DataFrame, abt_artifacts: dict) -> pd.DataFrame:
    """Recria features derivadas e alinha a entrada ao schema da ABT de treino."""
    x = pf.normalize_column_names(df)

    # A mesma regra de feature engineering usada na construção da ABT é reutilizada.
    x = pf.add_financial_ratios(x)

    # O schema oficial vem do abt_artifacts.pkl, e não do Streamlit.
    # Históricos inexistentes em uma nova solicitação recebem zero; demais campos
    # ausentes recebem NaN e serão imputados pelo preprocessor treinado.
    feature_columns = abt_artifacts["feature_columns"]
    for col in feature_columns:
        if col not in x.columns:
            x[col] = pf.default_value_for_missing_model_feature(col)

    return x[feature_columns].replace([np.inf, -np.inf], np.nan)


def transform_with_abt_artifacts(
    df: pd.DataFrame,
    abt_artifacts: dict,
):
    """Aplica o preprocessing treinado que está persistido em ``abt_artifacts.pkl``."""
    prepared = prepare_features(df, abt_artifacts)
    preprocessor = abt_artifacts["preprocessor"]
    transformed = preprocessor.transform(prepared)
    return prepared, transformed


def predict(
    records: Union[dict, list, pd.DataFrame],
    threshold: float | None = None,
    model_path: str | None = None,
    artifacts_path: str | None = None,
) -> pd.DataFrame:
    """Cruza ``abt_artifacts.pkl`` com ``model.pkl`` e retorna PD + decisão."""
    df_in = _as_dataframe(records)

    # Os dois artefatos são carregados separadamente e são obrigatórios.
    abt_artifacts = load_abt_artifacts(artifacts_path)
    model_bundle = load_model(model_path)

    # 1) abt_artifacts.pkl prepara e transforma os dados.
    _, X_transformed = transform_with_abt_artifacts(df_in, abt_artifacts)

    # 2) model.pkl recebe a matriz já transformada e calcula a probabilidade.
    estimator = model_bundle["model"]
    probability = estimator.predict_proba(X_transformed)[:, 1]

    decision_threshold = float(
        model_bundle.get("threshold", DECISION_THRESHOLD)
        if threshold is None
        else threshold
    )

    result = pd.DataFrame(index=df_in.index)
    if config.ID_COL in df_in.columns:
        result[config.ID_COL] = df_in[config.ID_COL].values
    result["PD_DEFAULT"] = probability
    result["THRESHOLD"] = decision_threshold
    result["CREDIT_DECISION"] = np.where(
        probability < decision_threshold,
        "APROVAR",
        "NEGAR / REVISAR",
    )
    result["ACTION_SUGGESTION"] = np.select(
        [
            probability < decision_threshold * 0.60,
            probability < decision_threshold,
            probability < min(1.0, decision_threshold + 0.15),
        ],
        [
            "Aprovação dentro da política definida.",
            "Aprovar com validação documental e limite conservador.",
            "Encaminhar para revisão manual de risco.",
        ],
        default="Não aprovar automaticamente; encaminhar para análise de risco.",
    )
    return result.reset_index(drop=True)


def score_dataframe(df: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    """Anexa PD e decisão ao DataFrame original usado pelo Streamlit."""
    scored = predict(df, threshold=threshold)
    base = df.reset_index(drop=True).copy()
    scored = scored.drop(columns=[config.ID_COL], errors="ignore")
    return pd.concat([base, scored], axis=1)


def score_sample(n: int = 1000, threshold: float | None = None) -> pd.DataFrame:
    """Seleciona uma amostra da ABT, calcula o score e salva um relatório demonstrativo."""
    df = storage.read_csv(config.ABT_PATH, low_memory=False)
    sample = df.sample(min(n, len(df)), random_state=config.RANDOM_STATE)
    scored = score_dataframe(sample, threshold=threshold)
    storage.write_csv(scored, "reports/scored_abt_sample.csv")
    return scored


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Score de risco de inadimplência Home Credit")
    parser.add_argument("--input", help="CSV de solicitações. Omitir para usar uma amostra da ABT.")
    parser.add_argument("--n", type=int, default=5, help="Quantidade de linhas")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    if args.input:
        source = pd.read_csv(args.input, nrows=args.n, low_memory=False)
        output = predict(source, threshold=args.threshold)
    else:
        output = score_sample(args.n, threshold=args.threshold)
    print(output.to_string(index=False))


if __name__ == "__main__":
    _cli()
