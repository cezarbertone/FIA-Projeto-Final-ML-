"""Funções reutilizáveis do pipeline e dos notebooks do projeto.

Este módulo concentra regras de transformação, validação, agregação, leitura de
artefatos e helpers de análise. Os scripts ``run()`` ficam responsáveis apenas por
orquestrar entrada/saída e gerar relatórios, enquanto os notebooks consomem estas
funções para evitar lógica duplicada ou valores/caminhos espalhados.

Organização do módulo:
1. I/O reutilizável;
2. limpeza e padronização;
3. agregações históricas;
4. construção da ABT e feature engineering;
5. helpers de EDA/qualidade;
6. helpers de avaliação e visualização.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from DataPipeline import config
from MLOps import storage


# -----------------------------------------------------------------------------
# 1. I/O reutilizável
# -----------------------------------------------------------------------------

# Este bloco promove um CSV colocado manualmente na Landing para o backend de armazenamento configurado, sem aplicar transformação de negócio.
# Usado em: DataPipeline/data_ingestion.py, na etapa que publica application_train, bureau e previous_application nas áreas raw do projeto.
def promote_landing_file(source: Path, destination: str, max_rows: int = 0) -> dict:
    """Promove um CSV da Landing para o backend configurado.

    ``max_rows`` existe apenas para testes rápidos. Com valor zero o arquivo é
    enviado sem ser carregado integralmente em memória.
    """
    if max_rows > 0:
        df = pd.read_csv(source, nrows=max_rows, low_memory=False)
        storage.write_csv(df, destination)
        rows = len(df)
    else:
        storage.upload_file(source, destination)
        rows = None

    return {
        "source": str(source),
        "destination": destination,
        "bytes": source.stat().st_size,
        "debug_rows": rows,
    }


# Este bloco centraliza a leitura de Dados/clean_data.csv para evitar que o caminho do arquivo fique repetido ou chumbado em outros códigos.
# Usado em: DataPipeline/abt_transform.py e DataPipeline/exp_analysis.ipynb.
def load_clean_data() -> pd.DataFrame:
    """Lê a base principal limpa."""
    return storage.read_csv(config.CLEAN_DATA_PATH, low_memory=False)


# Este bloco centraliza a leitura de Dados/abt.csv, que é a base analítica final entregue para a modelagem e para as análises pós-transformação.
# Usado em: DataPipeline/abt_overview.ipynb.
def load_abt() -> pd.DataFrame:
    """Lê a ABT utilizada na modelagem."""
    return storage.read_csv(config.ABT_PATH, low_memory=False)


# Este bloco lê de uma só vez as versões limpas de bureau e previous_application para análises que precisam enxergar as duas fontes históricas.
# Usado em: DataPipeline/exp_analysis.ipynb.
def load_clean_auxiliary_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lê as versões limpas de bureau e previous_application."""
    bureau = storage.read_csv(config.CLEAN_BUREAU_PATH, low_memory=False)
    previous = storage.read_csv(config.CLEAN_PREVIOUS_PATH, low_memory=False)
    return bureau, previous


# Este bloco lê o arquivo de métricas finais produzido pelo treinamento, mantendo o caminho desse artefato centralizado na configuração.
# Usado em: Model/evaluation.ipynb.
def load_metrics() -> dict:
    """Lê as métricas produzidas pelo treinamento."""
    return storage.read_json(config.METRICS_PATH)


# Este bloco converte o nome lógico de um relatório no caminho físico configurado e gera erro claro quando o nome informado não existe.
# Usado internamente por: load_report(), report_exists() e load_json_report(), que abastecem principalmente Model/evaluation.ipynb.
def report_path(report_name: str) -> str:
    """Resolve um nome lógico de relatório para seu caminho configurado."""
    try:
        return config.REPORT_PATHS[report_name]
    except KeyError as exc:
        known = ", ".join(sorted(config.REPORT_PATHS))
        raise KeyError(f"Relatório desconhecido: {report_name}. Opções: {known}") from exc


# Este bloco lê relatórios CSV gerados durante treinamento e avaliação sem espalhar os caminhos dos arquivos pelos notebooks.
# Usado em: Model/evaluation.ipynb para comparação de modelos, curva ROC, thresholds e relatórios de importância.
def load_report(report_name: str) -> pd.DataFrame:
    """Lê um relatório CSV gerado pelo pipeline/modelo."""
    return storage.read_csv(report_path(report_name), low_memory=False)


# Este bloco verifica se um relatório opcional foi realmente gerado antes que o notebook tente lê-lo, evitando quebra quando um artefato não existe.
# Usado em: Model/evaluation.ipynb para feature importance, permutation importance, SHAP e outros relatórios condicionais.
def report_exists(report_name: str) -> bool:
    """Indica se um relatório configurado já foi produzido."""
    return storage.exists(report_path(report_name))


# Este bloco lê relatórios em JSON usando o mesmo catálogo de caminhos centralizado dos demais artefatos do projeto.
# Usado em: Model/evaluation.ipynb, principalmente para consultar o status do SHAP quando esse relatório existir.
def load_json_report(report_name: str) -> dict:
    """Lê um relatório JSON configurado."""
    return storage.read_json(report_path(report_name))


# -----------------------------------------------------------------------------
# 2. Limpeza e padronização
# -----------------------------------------------------------------------------

# Este bloco padroniza os nomes das colunas em maiúsculas e remove espaços, garantindo o mesmo schema lógico entre ingestão, transformação e inferência.
# Usado em: funções de sanitização e agregação deste arquivo e diretamente em Model/predict.py durante a preparação de novas solicitações.
def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza nomes de colunas para maiúsculas e remove espaços externos."""
    result = df.copy()
    result.columns = [str(col).strip().upper() for col in result.columns]
    return result


# Este bloco limpa application_train, remove duplicidades por cliente e transforma o valor anômalo de DAYS_EMPLOYED em nulo sem alterar a granularidade da base.
# Usado em: DataPipeline/data_sanitization.py para gerar a base principal limpa Dados/clean_data.csv.
def sanitize_application_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Limpa ``application_train`` preservando sua granularidade."""
    result = normalize_column_names(df).drop_duplicates()
    if config.ID_COL in result.columns:
        result = result.drop_duplicates(subset=[config.ID_COL], keep="first")
    if "DAYS_EMPLOYED" in result.columns:
        result["DAYS_EMPLOYED"] = result["DAYS_EMPLOYED"].replace(
            config.DAYS_EMPLOYED_ANOMALY,
            np.nan,
        )
    return result


# Este bloco limpa bureau, remove duplicidades de créditos e impede valores negativos de dívida antes da criação das features históricas.
# Usado em: DataPipeline/data_sanitization.py para gerar a versão limpa de bureau.
def sanitize_bureau_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Limpa ``bureau`` e impede saldo de dívida negativo."""
    result = normalize_column_names(df).drop_duplicates()
    if "SK_ID_BUREAU" in result.columns:
        result = result.drop_duplicates(subset=["SK_ID_BUREAU"], keep="first")
    if "AMT_CREDIT_SUM_DEBT" in result.columns:
        result["AMT_CREDIT_SUM_DEBT"] = pd.to_numeric(
            result["AMT_CREDIT_SUM_DEBT"], errors="coerce"
        ).clip(lower=0)
    return result


# Este bloco limpa previous_application, remove duplicidades de solicitações anteriores e converte o sentinel 365243 das colunas de dias em nulo.
# Usado em: DataPipeline/data_sanitization.py para gerar a versão limpa de previous_application.
def sanitize_previous_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Limpa ``previous_application`` e trata o sentinel 365243 nas datas relativas."""
    result = normalize_column_names(df).drop_duplicates()
    if "SK_ID_PREV" in result.columns:
        result = result.drop_duplicates(subset=["SK_ID_PREV"], keep="first")
    for col in config.PREVIOUS_SENTINEL_COLUMNS:
        if col in result.columns:
            result[col] = result[col].replace(config.DAYS_EMPLOYED_ANOMALY, np.nan)
    return result


# -----------------------------------------------------------------------------
# 3. Agregações históricas
# -----------------------------------------------------------------------------

# Este bloco reduz bureau para uma linha por SK_ID_CURR e cria as features BUREAU_* que resumem quantidade de créditos, situação, dívida, atraso e exposição.
# Usado em: DataPipeline/feature_aggregation.py; o resultado é posteriormente incorporado à ABT por DataPipeline/abt_transform.py.
def aggregate_bureau_dataframe(bureau: pd.DataFrame) -> pd.DataFrame:
    """Agrega bureau para uma linha por ``SK_ID_CURR`` e cria features ``BUREAU_*``."""
    source = normalize_column_names(bureau)
    source = source[[c for c in config.BUREAU_AGG_SOURCE_COLUMNS if c in source.columns]].copy()

    if config.ID_COL not in source.columns or "SK_ID_BUREAU" not in source.columns:
        raise ValueError("bureau precisa conter SK_ID_CURR e SK_ID_BUREAU.")

    active = source["CREDIT_ACTIVE"] if "CREDIT_ACTIVE" in source.columns else pd.Series(
        index=source.index, dtype="object"
    )
    source["_active"] = active.eq("Active").astype("int8")
    source["_closed"] = active.eq("Closed").astype("int8")

    agg_spec: dict[str, tuple[str, str]] = {
        "BUREAU_CREDIT_COUNT": ("SK_ID_BUREAU", "count"),
        "BUREAU_ACTIVE_COUNT": ("_active", "sum"),
        "BUREAU_CLOSED_COUNT": ("_closed", "sum"),
    }
    optional = {
        "BUREAU_AMT_CREDIT_SUM_TOTAL": ("AMT_CREDIT_SUM", "sum"),
        "BUREAU_AMT_CREDIT_SUM_MEAN": ("AMT_CREDIT_SUM", "mean"),
        "BUREAU_AMT_DEBT_TOTAL": ("AMT_CREDIT_SUM_DEBT", "sum"),
        "BUREAU_DAY_OVERDUE_MAX": ("CREDIT_DAY_OVERDUE", "max"),
        "BUREAU_DAY_OVERDUE_MEAN": ("CREDIT_DAY_OVERDUE", "mean"),
        "BUREAU_AMT_OVERDUE_TOTAL": ("AMT_CREDIT_SUM_OVERDUE", "sum"),
        "BUREAU_DAYS_CREDIT_MIN": ("DAYS_CREDIT", "min"),
        "BUREAU_DAYS_CREDIT_MAX": ("DAYS_CREDIT", "max"),
        "BUREAU_DAYS_CREDIT_MEAN": ("DAYS_CREDIT", "mean"),
        "BUREAU_CNT_PROLONG_TOTAL": ("CNT_CREDIT_PROLONG", "sum"),
    }
    agg_spec.update({name: spec for name, spec in optional.items() if spec[0] in source.columns})

    result = source.groupby(config.ID_COL).agg(**agg_spec)
    result["BUREAU_ACTIVE_RATIO"] = (
        result["BUREAU_ACTIVE_COUNT"] / result["BUREAU_CREDIT_COUNT"].replace(0, np.nan)
    )
    if {"BUREAU_AMT_DEBT_TOTAL", "BUREAU_AMT_CREDIT_SUM_TOTAL"}.issubset(result.columns):
        result["BUREAU_DEBT_CREDIT_RATIO"] = (
            result["BUREAU_AMT_DEBT_TOTAL"]
            / result["BUREAU_AMT_CREDIT_SUM_TOTAL"].replace(0, np.nan)
        )
    return result.replace([np.inf, -np.inf], np.nan).reset_index()


# Este bloco reduz previous_application para uma linha por SK_ID_CURR e cria as features PREV_* de quantidade, aprovação, recusa, valores e histórico das propostas.
# Usado em: DataPipeline/feature_aggregation.py; o resultado é posteriormente incorporado à ABT por DataPipeline/abt_transform.py.
def aggregate_previous_dataframe(previous: pd.DataFrame) -> pd.DataFrame:
    """Agrega previous_application para uma linha por cliente e cria ``PREV_*``."""
    source = normalize_column_names(previous)
    source = source[[c for c in config.PREVIOUS_AGG_SOURCE_COLUMNS if c in source.columns]].copy()

    if config.ID_COL not in source.columns or "SK_ID_PREV" not in source.columns:
        raise ValueError("previous_application precisa conter SK_ID_CURR e SK_ID_PREV.")

    status = source["NAME_CONTRACT_STATUS"] if "NAME_CONTRACT_STATUS" in source.columns else pd.Series(
        index=source.index, dtype="object"
    )
    source["_approved"] = status.eq("Approved").astype("int8")
    source["_refused"] = status.eq("Refused").astype("int8")

    if {"AMT_CREDIT", "AMT_APPLICATION"}.issubset(source.columns):
        source["_credit_app_ratio"] = (
            source["AMT_CREDIT"] / source["AMT_APPLICATION"].replace(0, np.nan)
        )

    agg_spec: dict[str, tuple[str, str]] = {
        "PREV_APP_COUNT": ("SK_ID_PREV", "count"),
        "PREV_APPROVED_COUNT": ("_approved", "sum"),
        "PREV_REFUSED_COUNT": ("_refused", "sum"),
    }
    optional = {
        "PREV_AMT_CREDIT_MEAN": ("AMT_CREDIT", "mean"),
        "PREV_AMT_CREDIT_TOTAL": ("AMT_CREDIT", "sum"),
        "PREV_AMT_APPLICATION_MEAN": ("AMT_APPLICATION", "mean"),
        "PREV_CREDIT_APP_RATIO_MEAN": ("_credit_app_ratio", "mean"),
        "PREV_AMT_DOWN_PAYMENT_MEAN": ("AMT_DOWN_PAYMENT", "mean"),
        "PREV_DAYS_DECISION_MAX": ("DAYS_DECISION", "max"),
        "PREV_DAYS_DECISION_MIN": ("DAYS_DECISION", "min"),
        "PREV_CNT_PAYMENT_MEAN": ("CNT_PAYMENT", "mean"),
    }
    agg_spec.update({name: spec for name, spec in optional.items() if spec[0] in source.columns})

    result = source.groupby(config.ID_COL).agg(**agg_spec)
    result["PREV_APPROVAL_RATE"] = (
        result["PREV_APPROVED_COUNT"] / result["PREV_APP_COUNT"].replace(0, np.nan)
    )
    result["PREV_REFUSED_RATE"] = (
        result["PREV_REFUSED_COUNT"] / result["PREV_APP_COUNT"].replace(0, np.nan)
    )
    return result.replace([np.inf, -np.inf], np.nan).reset_index()


# -----------------------------------------------------------------------------
# 4. ABT e feature engineering
# -----------------------------------------------------------------------------

# Este bloco define como preencher uma feature esperada pelo modelo quando ela não chega na inferência: histórico recebe zero e demais campos recebem NaN.
# Usado em: Model/predict.py para alinhar uma nova solicitação ao mesmo conjunto de features utilizado no treinamento.
def default_value_for_missing_model_feature(column: str):
    """Define o valor padrão de uma feature ausente na inferência.

    Features históricas inexistentes recebem zero; demais colunas recebem ``NaN``
    para que o Pipeline treinado aplique a imputação aprendida no treino.
    """
    return 0 if column.startswith(config.HISTORY_PREFIXES) else np.nan


# Este bloco cria CREDIT_INCOME_RATIO, ANNUITY_INCOME_RATIO e ANNUITY_CREDIT_RATIO usando uma única regra compartilhada entre preparação da ABT e predição.
# Usado em: build_abt_dataframe() deste arquivo, Model/predict.py e DataPipeline/exp_analysis.ipynb.
def add_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Cria as três razões financeiras usadas na ABT e na inferência."""
    result = df.copy()
    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(result.columns):
        result["CREDIT_INCOME_RATIO"] = (
            result["AMT_CREDIT"] / result["AMT_INCOME_TOTAL"].replace(0, np.nan)
        )
    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(result.columns):
        result["ANNUITY_INCOME_RATIO"] = (
            result["AMT_ANNUITY"] / result["AMT_INCOME_TOTAL"].replace(0, np.nan)
        )
    if {"AMT_ANNUITY", "AMT_CREDIT"}.issubset(result.columns):
        result["ANNUITY_CREDIT_RATIO"] = (
            result["AMT_ANNUITY"] / result["AMT_CREDIT"].replace(0, np.nan)
        )
    return result.replace([np.inf, -np.inf], np.nan)


# Este bloco faz os joins das features agregadas de Bureau e Previous na aplicação principal por SK_ID_CURR e preenche ausência de histórico com zero.
# Usado internamente por: build_abt_dataframe(), que é chamado por DataPipeline/abt_transform.py.
def merge_historical_features(
    application: pd.DataFrame,
    bureau_features: pd.DataFrame | None = None,
    previous_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Une features históricas já agregadas sem alterar a granularidade da aplicação."""
    result = application.copy()
    for features in (bureau_features, previous_features):
        if features is not None and not features.empty:
            result = result.merge(features, on=config.ID_COL, how="left")

    history_cols = [c for c in result.columns if c.startswith(config.HISTORY_PREFIXES)]
    if history_cols:
        result[history_cols] = result[history_cols].fillna(0)
    return result


# Este bloco identifica colunas com percentual de nulos acima do limite definido em config.py, preservando ID, TARGET e colunas explicitamente protegidas.
# Usado internamente por: build_abt_dataframe(), durante a criação da ABT executada em DataPipeline/abt_transform.py.
def columns_above_missing_threshold(
    df: pd.DataFrame,
    threshold: float = config.NULL_DROP_THRESHOLD,
    protected_columns: Iterable[str] | None = None,
) -> list[str]:
    """Lista colunas que ultrapassam o limite de nulos e podem ser removidas."""
    protected = set(protected_columns or config.PROTECTED_NULL_COLUMNS)
    protected.update({config.ID_COL, config.TARGET_COL})
    rates = df.isna().mean()
    return [col for col, rate in rates.items() if rate > threshold and col not in protected]


# Este bloco monta a ABT completa: cria ratios, une históricos, remove colunas excessivamente nulas, trata infinitos e garante uma linha por SK_ID_CURR.
# Usado em: DataPipeline/abt_transform.py para gerar Dados/abt.csv, que será consumido pelo treinamento do modelo.
def build_abt_dataframe(
    application: pd.DataFrame,
    bureau_features: pd.DataFrame | None = None,
    previous_features: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Constrói a ABT a partir das fontes já limpas/agregadas."""
    result = add_financial_ratios(application)
    result = merge_historical_features(result, bureau_features, previous_features)
    dropped = columns_above_missing_threshold(result)
    result = result.drop(columns=dropped).replace([np.inf, -np.inf], np.nan)
    if config.ID_COL in result.columns:
        result = result.drop_duplicates(subset=[config.ID_COL], keep="first")
    return result, dropped


# -----------------------------------------------------------------------------
# 5. Helpers de EDA e qualidade
# -----------------------------------------------------------------------------

# Este bloco resume rapidamente quantidade de linhas, colunas, IDs únicos, IDs duplicados e taxa de inadimplência da base analisada.
# Usado em: DataPipeline/exp_analysis.ipynb e DataPipeline/abt_overview.ipynb.
def dataset_overview(df: pd.DataFrame) -> dict:
    """Resume shape, unicidade do ID e taxa do TARGET."""
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "unique_ids": int(df[config.ID_COL].nunique()) if config.ID_COL in df.columns else None,
        "duplicate_ids": int(df[config.ID_COL].duplicated().sum()) if config.ID_COL in df.columns else None,
        "target_rate": float(df[config.TARGET_COL].mean()) if config.TARGET_COL in df.columns else None,
    }


# Este bloco calcula quantidade e participação percentual de TARGET=0 e TARGET=1 para evidenciar o balanceamento das classes.
# Usado em: DataPipeline/exp_analysis.ipynb.
def target_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna quantidade e percentual das classes do TARGET."""
    counts = df[config.TARGET_COL].value_counts().sort_index()
    return pd.DataFrame({"quantidade": counts, "percentual": counts / counts.sum()})


# Este bloco calcula o percentual de valores nulos de cada coluna e ordena do maior para o menor para apoiar a análise de qualidade dos dados.
# Usado em: DataPipeline/exp_analysis.ipynb.
def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna percentual de nulos por coluna em ordem decrescente."""
    return (df.isna().mean() * 100).sort_values(ascending=False).to_frame("missing_pct")


# Este bloco gera estatísticas descritivas apenas para as colunas solicitadas que realmente existem na base, evitando erro quando alguma variável não está presente.
# Usado em: DataPipeline/exp_analysis.ipynb.
def describe_available_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    percentiles: Iterable[float] = (.01, .05, .25, .50, .75, .95, .99),
) -> pd.DataFrame:
    """Descreve somente as colunas solicitadas que existem no DataFrame."""
    available = [c for c in columns if c in df.columns]
    if not available:
        return pd.DataFrame()
    return df[available].describe(percentiles=list(percentiles)).T


# Este bloco converte DAYS_BIRTH em idade aproximada em anos e cria faixas etárias exclusivamente para análise exploratória.
# Usado internamente por: age_target_summary(), que aparece em DataPipeline/exp_analysis.ipynb.
def add_age_analysis_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria idade em anos e faixa etária para uso exploratório."""
    result = df.copy()
    if "DAYS_BIRTH" not in result.columns:
        return result
    result["AGE_YEARS"] = -pd.to_numeric(result["DAYS_BIRTH"], errors="coerce") / 365.25
    result["AGE_BUCKET"] = pd.cut(
        result["AGE_YEARS"],
        bins=config.AGE_BUCKET_BINS,
        right=False,
    )
    return result


# Este bloco agrupa os clientes por faixa etária e calcula volume e taxa média de inadimplência em cada grupo.
# Usado em: DataPipeline/exp_analysis.ipynb.
def age_target_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula volume e taxa de inadimplência por faixa etária."""
    analysis = add_age_analysis_features(df)
    if "AGE_BUCKET" not in analysis.columns:
        return pd.DataFrame()
    result = analysis.groupby("AGE_BUCKET", observed=False)[config.TARGET_COL].agg(["count", "mean"])
    return result.rename(columns={"mean": "default_rate"})


# Este bloco compara as médias de variáveis numéricas entre TARGET=0 e TARGET=1 para identificar diferenças de comportamento entre as classes.
# Usado em: DataPipeline/exp_analysis.ipynb e DataPipeline/abt_overview.ipynb.
def target_group_summary(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Compara média de variáveis disponíveis entre as classes do TARGET."""
    available = [c for c in columns if c in df.columns]
    if not available:
        return pd.DataFrame()
    return df.groupby(config.TARGET_COL)[available].mean(numeric_only=True).T


# Este bloco resume média, mediana e quantidade válida dos EXT_SOURCE_* separando os resultados por TARGET.
# Usado em: DataPipeline/exp_analysis.ipynb.
def external_source_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compara média, mediana e contagem válida dos EXT_SOURCE_* por TARGET."""
    columns = [c for c in config.EXT_SOURCE_COLUMNS if c in df.columns]
    if not columns:
        return pd.DataFrame()
    return df.groupby(config.TARGET_COL)[columns].agg(["mean", "median", "count"]).T


# Este bloco calcula volume e taxa de inadimplência para as categorias mais frequentes de uma variável categórica informada.
# Usado em: DataPipeline/exp_analysis.ipynb.
def categorical_target_summary(df: pd.DataFrame, column: str, top_n: int = 15) -> pd.DataFrame:
    """Resume volume e taxa de inadimplência das categorias mais frequentes."""
    if column not in df.columns:
        return pd.DataFrame()
    summary = (
        df.groupby(column, dropna=False)[config.TARGET_COL]
        .agg(["count", "mean"])
        .rename(columns={"mean": "default_rate"})
        .sort_values("count", ascending=False)
    )
    return summary.head(top_n)


# Este bloco resume tamanho, quantidade de clientes únicos e média de registros por cliente nas fontes bureau e previous_application.
# Usado em: DataPipeline/exp_analysis.ipynb para explicar a granularidade das fontes auxiliares.
def auxiliary_sources_summary(
    bureau: pd.DataFrame,
    previous: pd.DataFrame,
) -> pd.DataFrame:
    """Resume volume, clientes cobertos e granularidade das fontes auxiliares."""
    return pd.DataFrame({
        "fonte": ["bureau", "previous_application"],
        "linhas": [len(bureau), len(previous)],
        "clientes_unicos": [
            bureau[config.ID_COL].nunique() if config.ID_COL in bureau.columns else np.nan,
            previous[config.ID_COL].nunique() if config.ID_COL in previous.columns else np.nan,
        ],
        "linhas_por_cliente_media": [
            len(bureau) / max(1, bureau[config.ID_COL].nunique()) if config.ID_COL in bureau.columns else np.nan,
            len(previous) / max(1, previous[config.ID_COL].nunique()) if config.ID_COL in previous.columns else np.nan,
        ],
    })


# Este bloco separa as colunas da ABT por origem analítica: identificador/alvo, Application, Bureau, Previous e razões financeiras.
# Usado internamente por: feature_block_table(), que aparece em DataPipeline/abt_overview.ipynb.
def abt_feature_blocks(abt: pd.DataFrame) -> Mapping[str, list[str]]:
    """Classifica as colunas da ABT por origem analítica."""
    id_target = [c for c in abt.columns if c in {config.ID_COL, config.TARGET_COL}]
    bureau = [c for c in abt.columns if c.startswith("BUREAU_")]
    previous = [c for c in abt.columns if c.startswith("PREV_")]
    ratios = [c for c in config.RATIO_COLUMNS if c in abt.columns]
    reserved = set(id_target + bureau + previous + ratios)
    application = [c for c in abt.columns if c not in reserved]
    return {
        "Identificador/Alvo": id_target,
        "Application": application,
        "Bureau": bureau,
        "Previous": previous,
        "Razões": ratios,
    }


# Este bloco transforma a classificação das features da ABT em uma tabela com a quantidade de colunas de cada bloco de origem.
# Usado em: DataPipeline/abt_overview.ipynb.
def feature_block_table(abt: pd.DataFrame) -> pd.DataFrame:
    """Conta quantas colunas pertencem a cada bloco da ABT."""
    blocks = abt_feature_blocks(abt)
    return pd.DataFrame({
        "bloco": list(blocks.keys()),
        "quantidade": [len(values) for values in blocks.values()],
    })


# Este bloco devolve listas separadas das features BUREAU_* e PREV_* existentes na ABT para facilitar inspeção e documentação.
# Usado em: DataPipeline/abt_overview.ipynb.
def historical_feature_lists(abt: pd.DataFrame) -> dict[str, list[str]]:
    """Retorna listas de features históricas por prefixo."""
    return {
        "Bureau": [c for c in abt.columns if c.startswith("BUREAU_")],
        "Previous": [c for c in abt.columns if c.startswith("PREV_")],
    }


# Este bloco calcula qual percentual dos clientes da ABT possui histórico de Bureau e qual percentual possui aplicações anteriores.
# Usado em: DataPipeline/abt_overview.ipynb.
def history_coverage(abt: pd.DataFrame) -> pd.DataFrame:
    """Calcula cobertura de Bureau e Previous na população da ABT."""
    values: dict[str, float] = {}
    if "BUREAU_CREDIT_COUNT" in abt.columns:
        values["Bureau"] = float((abt["BUREAU_CREDIT_COUNT"] > 0).mean())
    if "PREV_APP_COUNT" in abt.columns:
        values["Previous"] = float((abt["PREV_APP_COUNT"] > 0).mean())
    return pd.Series(values, name="coverage", dtype="float64").to_frame()


# Este bloco cria uma visão de qualidade por coluna contendo tipo de dado, percentual de nulos e cardinalidade.
# Usado em: DataPipeline/abt_overview.ipynb.
def quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Resume tipo, taxa de nulos e cardinalidade de todas as colunas."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_pct": df.isna().mean() * 100,
        "n_unique": df.nunique(dropna=False),
    }).sort_values("missing_pct", ascending=False)


# Este bloco calcula a correlação linear das variáveis numéricas com TARGET e ordena as features pela força absoluta dessa relação.
# Usado em: DataPipeline/abt_overview.ipynb.
def numeric_target_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula correlações numéricas com o TARGET ordenadas pelo valor absoluto."""
    numeric = df.select_dtypes(include=np.number)
    if config.TARGET_COL not in numeric.columns:
        return pd.DataFrame(columns=["corr_target"])
    corr = (
        numeric.corr(numeric_only=True)[config.TARGET_COL]
        .drop(config.TARGET_COL)
        .sort_values(key=lambda series: series.abs(), ascending=False)
    )
    return corr.to_frame("corr_target")


# -----------------------------------------------------------------------------
# 6. Helpers de avaliação e visualização dos notebooks
# -----------------------------------------------------------------------------

# Este bloco organiza as principais métricas do holdout em uma estrutura legível para apresentação, incluindo AUC, KS, Recall, Precision, F1 e threshold.
# Usado em: Model/evaluation.ipynb.
def model_metrics_summary(metrics: dict) -> pd.Series:
    """Transforma o bloco de métricas finais em uma série legível."""
    hold = metrics["holdout"]
    return pd.Series({
        "Modelo vencedor": metrics["best_model"],
        "AUC-ROC": hold["auc_roc"],
        "KS": hold["ks"],
        "Average Precision": hold["average_precision"],
        "Recall inadimplente": hold["recall_default"],
        "Precision inadimplente": hold["precision_default"],
        "F1 inadimplente": hold["f1_default"],
        "Accuracy": hold["accuracy"],
        "Threshold": hold["threshold"],
    })


# Este bloco reconstrói a matriz de confusão 2x2 a partir dos valores armazenados em metrics.json.
# Usado em: Model/evaluation.ipynb.
def confusion_matrix_from_metrics(metrics: dict) -> np.ndarray:
    """Reconstrói a matriz 2x2 a partir de ``metrics.json``."""
    cm = metrics["holdout"]["confusion_matrix"]
    return np.array([
        [cm["true_negative"], cm["false_positive"]],
        [cm["false_negative"], cm["true_positive"]],
    ])


# Este bloco gera o gráfico da taxa de inadimplência por faixa etária a partir do resumo já calculado.
# Usado em: DataPipeline/exp_analysis.ipynb.
def plot_age_target_summary(summary: pd.DataFrame) -> None:
    """Plota taxa de inadimplência por faixa etária."""
    import matplotlib.pyplot as plt

    if summary.empty or "default_rate" not in summary.columns:
        return
    summary["default_rate"].mul(100).plot(marker="o", title="Taxa de inadimplência por faixa etária (%)")
    plt.ylabel("%")
    plt.show()


# Este bloco gera um gráfico com a quantidade de features provenientes de cada bloco analítico da ABT.
# Usado em: DataPipeline/abt_overview.ipynb.
def plot_feature_block_counts(table: pd.DataFrame) -> None:
    """Plota a quantidade de colunas em cada bloco da ABT."""
    import matplotlib.pyplot as plt

    if table.empty:
        return
    table.set_index("bloco")["quantidade"].plot(kind="bar", title="Colunas por bloco")
    plt.ylabel("nº de colunas")
    plt.show()


# Este bloco gera o gráfico percentual de TARGET=0 e TARGET=1 para visualizar o desbalanceamento da variável alvo.
# Usado em: DataPipeline/exp_analysis.ipynb.
def plot_target_distribution(table: pd.DataFrame) -> None:
    """Plota a participação percentual de TARGET=0/1."""
    import matplotlib.pyplot as plt

    ax = (table["percentual"] * 100).plot(kind="bar", title="Distribuição do TARGET (%)")
    ax.set_xlabel("TARGET (0=adimplente, 1=inadimplente)")
    ax.set_ylabel("% da base")
    plt.show()


# Este bloco compara visualmente a distribuição de variáveis contínuas entre adimplentes e inadimplentes por meio de curvas de densidade.
# Usado em: DataPipeline/exp_analysis.ipynb.
def plot_density_by_target(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Plota densidade de cada coluna separada pelo TARGET."""
    import matplotlib.pyplot as plt

    for col in [c for c in columns if c in df.columns]:
        df.groupby(config.TARGET_COL)[col].plot(kind="density", legend=True, title=f"{col} por TARGET")
        plt.xlim(0, 1)
        plt.show()


# Este bloco transforma um resumo categórico em gráfico horizontal de taxa de default para facilitar a comparação entre categorias.
# Uso atual: função auxiliar disponível para DataPipeline/exp_analysis.ipynb, mas não é chamada diretamente na versão atual do notebook.
def plot_target_rate_by_category(summary: pd.DataFrame, column: str) -> None:
    """Plota a taxa de default de uma tabela categórica já resumida."""
    import matplotlib.pyplot as plt

    if summary.empty:
        return
    summary.sort_values("default_rate")["default_rate"].plot(
        kind="barh", title=f"Taxa de default — {column}"
    )
    plt.xlabel("Taxa de default")
    plt.show()


# Este bloco plota as maiores correlações lineares com TARGET, limitando a visualização às features mais relevantes.
# Usado em: DataPipeline/abt_overview.ipynb.
def plot_target_correlations(correlations: pd.DataFrame, top_n: int = 20) -> None:
    """Plota as correlações lineares mais relevantes com o TARGET."""
    import matplotlib.pyplot as plt

    if correlations.empty:
        return
    correlations.head(top_n)["corr_target"].sort_values().plot(
        kind="barh", title="Top correlações lineares com TARGET"
    )
    plt.show()


# Este bloco compara as razões financeiras entre as classes de TARGET por boxplots e limita valores extremos ao percentil 99 para melhorar a leitura.
# Usado em: DataPipeline/abt_overview.ipynb.
def plot_ratio_boxplots(df: pd.DataFrame) -> None:
    """Plota as razões financeiras por TARGET limitando a visualização ao p99."""
    import matplotlib.pyplot as plt

    for col in [c for c in config.RATIO_COLUMNS if c in df.columns]:
        p99 = df[col].quantile(.99)
        df[df[col] <= p99].boxplot(column=col, by=config.TARGET_COL)
        plt.suptitle("")
        plt.title(f"{col} por TARGET (até p99)")
        plt.show()


# Este bloco plota a CV-AUC média dos algoritmos candidatos para mostrar visualmente qual modelo teve melhor desempenho na etapa de seleção.
# Usado em: Model/evaluation.ipynb.
def plot_model_comparison(comparison: pd.DataFrame) -> None:
    """Plota a CV-AUC média dos candidatos."""
    import matplotlib.pyplot as plt

    comparison.set_index("model")["cv_auc_mean"].plot(
        kind="bar", ylim=(.5, 1), title="CV-AUC média por algoritmo"
    )
    plt.ylabel("AUC")
    plt.show()


# Este bloco plota a curva ROC do holdout e exibe a AUC do modelo vencedor, permitindo avaliar a capacidade de separação entre as classes.
# Usado em: Model/evaluation.ipynb.
def plot_roc_curve(roc: pd.DataFrame, auc: float) -> None:
    """Plota a curva ROC salva no holdout."""
    import matplotlib.pyplot as plt

    plt.plot(roc["fpr"], roc["tpr"], label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Curva ROC — holdout")
    plt.legend()
    plt.show()


# Este bloco exibe a matriz de confusão com rótulos de negócio, mostrando acertos e erros para clientes adimplentes e inadimplentes.
# Usado em: Model/evaluation.ipynb.
def plot_confusion_matrix(matrix: np.ndarray) -> None:
    """Plota matriz de confusão 2x2 com rótulos de negócio."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center", fontsize=13)
    ax.set_xticks([0, 1], labels=["Pred. adimplente", "Pred. inadimplente"])
    ax.set_yticks([0, 1], labels=["Real adimplente", "Real inadimplente"])
    ax.set_title("Matriz de confusão — holdout")
    plt.show()


# Este bloco mostra como diferentes thresholds alteram taxa de aprovação, Recall, Precision e falso negativo, conectando performance técnica à política de crédito.
# Usado em: Model/evaluation.ipynb.
def plot_threshold_tradeoff(thresholds: pd.DataFrame) -> None:
    """Plota o trade-off entre aprovação e métricas por threshold."""
    import matplotlib.pyplot as plt

    columns = [
        "approval_rate",
        "recall_default",
        "precision_default",
        "false_negative_rate",
    ]
    thresholds.set_index("threshold")[columns].plot(marker="o", title="Impacto do threshold")
    plt.ylim(0, 1)
    plt.show()


# Este bloco cria um gráfico horizontal para relatórios ranqueados de importância de features, recebendo o nome da métrica que deve ser exibida no eixo.
# Usado em: Model/evaluation.ipynb para feature importance, permutation importance e SHAP quando os respectivos relatórios existem.
def plot_ranked_report(
    data: pd.DataFrame,
    x_column: str,
    title: str,
    top_n: int = 20,
) -> None:
    """Exibe graficamente um relatório de importância já ordenado."""
    import matplotlib.pyplot as plt

    if data.empty or x_column not in data.columns or "feature" not in data.columns:
        return
    data.head(top_n).sort_values(x_column).plot.barh(
        x="feature", y=x_column, legend=False, title=title
    )
    plt.show()
