"""Treinamento do modelo de risco de inadimplência.

Fluxo:
1. separa treino (80%) e holdout (20%) de forma estratificada;
2. compara Logistic Regression, Random Forest e Gradient Boosting por CV-AUC;
3. executa GridSearchCV somente no algoritmo vencedor;
4. treina o vencedor ajustado em todo o conjunto de treino;
5. avalia uma única vez no holdout;
6. separa e salva os artefatos finais de produção em dois arquivos:
   - ``DataPipeline/abt_artifacts.pkl``: preprocessing + contrato da ABT;
   - ``Model/model.pkl``: estimador vencedor + threshold;
7. salva métricas e relatórios de avaliação/explicabilidade.

Durante seleção, tuning e treino final, imputação, encoding e scaling ficam dentro
do Pipeline do Scikit-Learn e são aprendidos somente nos dados de treino de cada fit.
Na persistência final, o preprocessor ajustado é separado do estimador para deixar
explícito o cruzamento ``abt_artifacts.pkl + model.pkl`` durante a inferência.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from MLOps import storage

_cfg_path = Path(__file__).with_name("config.py")
_spec = importlib.util.spec_from_file_location("model_config", _cfg_path)
config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(config)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)
NEEDS_SAMPLE_WEIGHT = {"gradient_boosting"}


def build_preprocessors(X: pd.DataFrame):
    """Separa colunas e monta preprocessing numérico/categórico dentro do Pipeline."""
    numeric = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    linear_numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    tree_numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    linear = ColumnTransformer([
        ("numeric", linear_numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ], remainder="drop")
    tree = ColumnTransformer([
        ("numeric", tree_numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ], remainder="drop")
    return linear, tree, numeric, categorical


def build_candidates(linear_preprocessor, tree_preprocessor):
    """Monta os três candidatos usando o mesmo contrato de entrada da ABT."""
    return {
        "logistic_regression": Pipeline([
            ("preprocessor", linear_preprocessor),
            ("model", LogisticRegression(**config.LOGISTIC_PARAMS)),
        ]),
        "random_forest": Pipeline([
            ("preprocessor", tree_preprocessor),
            ("model", RandomForestClassifier(**config.RANDOM_FOREST_PARAMS)),
        ]),
        "gradient_boosting": Pipeline([
            ("preprocessor", tree_preprocessor),
            ("model", GradientBoostingClassifier(**config.GRADIENT_BOOSTING_PARAMS)),
        ]),
    }


def fit_model(model, name: str, X: pd.DataFrame, y: pd.Series):
    """Treina um candidato; Gradient Boosting recebe sample_weight balanceado."""
    if name in NEEDS_SAMPLE_WEIGHT:
        weights = compute_sample_weight(class_weight="balanced", y=y)
        model.fit(X, y, model__sample_weight=weights)
    else:
        model.fit(X, y)
    return model


def fit_params(name: str, y: pd.Series) -> dict:
    """Parâmetros adicionais usados pelo GridSearchCV no fit do candidato."""
    if name in NEEDS_SAMPLE_WEIGHT:
        return {"model__sample_weight": compute_sample_weight(class_weight="balanced", y=y)}
    return {}


def ks_statistic(y_true, y_score) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def confusion_metrics(y_true, y_score, threshold: float) -> dict:
    pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "approval_rate": float((pred == 0).mean()),
        "review_or_deny_rate": float((pred == 1).mean()),
        "recall_default": float(recall_score(y_true, pred, zero_division=0)),
        "precision_default": float(precision_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
    }


def subsample(X, y, frac):
    """Cria subamostra estratificada somente para seleção e busca de hiperparâmetros."""
    if not frac or frac >= 1:
        return X, y
    Xs, _, ys, _ = train_test_split(
        X, y, train_size=frac, stratify=y, random_state=config.RANDOM_STATE
    )
    return Xs, ys


def cross_validate_candidate(name, model, X, y, cv):
    """Executa CV manual para aplicar corretamente sample_weight ao Gradient Boosting."""
    scores = []
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        fitted = fit_model(clone(model), name, X.iloc[train_idx], y.iloc[train_idx])
        proba = fitted.predict_proba(X.iloc[valid_idx])[:, 1]
        auc = float(roc_auc_score(y.iloc[valid_idx], proba))
        scores.append(auc)
        LOGGER.info("%s fold %s/%s: AUC=%.4f", name, fold, config.CV_FOLDS, auc)
    return scores


def evaluate(model, X_hold, y_hold, threshold):
    """Calcula as métricas finais usando somente o holdout."""
    proba = model.predict_proba(X_hold)[:, 1]
    pred = (proba >= threshold).astype(int)
    return proba, {
        "rows": int(len(X_hold)),
        "default_rate": float(y_hold.mean()),
        "auc_roc": float(roc_auc_score(y_hold, proba)),
        "ks": ks_statistic(y_hold, proba),
        "average_precision": float(average_precision_score(y_hold, proba)),
        "recall_default": float(recall_score(y_hold, pred, zero_division=0)),
        "precision_default": float(precision_score(y_hold, pred, zero_division=0)),
        "f1_default": float(f1_score(y_hold, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_hold, pred)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_metrics(y_hold, proba, threshold),
    }


def transformed_feature_names(pipe) -> list[str]:
    """Recupera os nomes das features após o ColumnTransformer."""
    try:
        return pipe.named_steps["preprocessor"].get_feature_names_out().tolist()
    except Exception:
        return []


def save_explainability(pipe, X_hold, y_hold):
    """Gera importância nativa, permutation importance e SHAP global quando disponível."""
    names = transformed_feature_names(pipe)
    estimator = pipe.named_steps["model"]

    values = getattr(estimator, "feature_importances_", None)
    if values is None and hasattr(estimator, "coef_"):
        values = np.ravel(estimator.coef_)
    if values is not None:
        n = min(len(names), len(values))
        native = pd.DataFrame({"feature": names[:n], "importance": np.asarray(values)[:n]})
        native["abs_value"] = native["importance"].abs()
        storage.write_csv(native.sort_values("abs_value", ascending=False), "reports/feature_importance.csv")

    X_perm, y_perm = X_hold, y_hold
    if config.PERMUTATION_SAMPLE_ROWS and len(X_hold) > config.PERMUTATION_SAMPLE_ROWS:
        X_perm, _, y_perm, _ = train_test_split(
            X_hold,
            y_hold,
            train_size=config.PERMUTATION_SAMPLE_ROWS,
            stratify=y_hold,
            random_state=config.RANDOM_STATE,
        )
    result = permutation_importance(
        pipe,
        X_perm,
        y_perm,
        n_repeats=config.PERMUTATION_REPEATS,
        scoring="roc_auc",
        random_state=config.RANDOM_STATE,
        n_jobs=1,
    )
    perm = pd.DataFrame({
        "feature": X_perm.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)
    storage.write_csv(perm, "reports/permutation_importance.csv")

    status = {"enabled": config.ENABLE_SHAP, "status": "not_run"}
    if config.ENABLE_SHAP:
        try:
            import shap
            sample = X_hold.sample(min(config.SHAP_SAMPLE_ROWS, len(X_hold)), random_state=config.RANDOM_STATE)
            transformed = pipe.named_steps["preprocessor"].transform(sample)
            explainer = shap.Explainer(estimator, transformed, feature_names=names)
            arr = np.asarray(explainer(transformed).values)
            if arr.ndim == 3:
                arr = arr[:, :, -1]
            shap_df = pd.DataFrame({
                "feature": names[:arr.shape[1]],
                "mean_abs_shap": np.abs(arr).mean(axis=0),
            }).sort_values("mean_abs_shap", ascending=False)
            storage.write_csv(shap_df, "reports/shap_importance.csv")
            status = {"enabled": True, "status": "ok", "sample_rows": int(len(sample))}
        except Exception as exc:
            status = {"enabled": True, "status": "failed", "error": str(exc)}
    storage.write_json(status, "reports/shap_status.json")


def train():
    LOGGER.info("Carregando ABT: %s", config.ABT_PATH)
    df = storage.read_csv(config.ABT_PATH, low_memory=False)
    if config.TARGET_COL not in df.columns:
        raise ValueError(f"ABT sem a coluna alvo {config.TARGET_COL}.")

    X = df.drop(columns=[c for c in (config.ID_COL, config.TARGET_COL) if c in df.columns])
    y = df[config.TARGET_COL].astype(int)

    X_train, X_hold, y_train, y_hold = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )
    X_select, y_select = subsample(X_train, y_train, config.SEARCH_SAMPLE_FRAC)
    LOGGER.info("Amostras=%s | Features=%s | Default=%.2f%%", len(df), X.shape[1], y.mean() * 100)
    LOGGER.info("Treino=%s | Holdout=%s | Seleção/Busca=%s", len(X_train), len(X_hold), len(X_select))

    linear_pre, tree_pre, numeric, categorical = build_preprocessors(X_train)
    candidates = build_candidates(linear_pre, tree_pre)
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    comparison = []
    for name, model in candidates.items():
        fold_scores = cross_validate_candidate(name, model, X_select, y_select, cv)
        comparison.append({
            "model": name,
            "cv_auc_mean": float(np.mean(fold_scores)),
            "cv_auc_std": float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0,
            "folds": config.CV_FOLDS,
            "fold_scores": ";".join(f"{value:.6f}" for value in fold_scores),
        })
    comparison_df = pd.DataFrame(comparison).sort_values("cv_auc_mean", ascending=False)
    storage.write_csv(comparison_df, "reports/model_comparison.csv")
    winner = str(comparison_df.iloc[0]["model"])
    LOGGER.info("Vencedor por CV-AUC: %s", winner)

    search = GridSearchCV(
        candidates[winner],
        config.SEARCH_GRIDS[winner],
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        return_train_score=True,
    )
    search.fit(X_select, y_select, **fit_params(winner, y_select))
    storage.write_csv(pd.DataFrame(search.cv_results_), "reports/grid_search_results.csv")
    LOGGER.info("GridSearch: best_score=%.4f | best_params=%s", search.best_score_, search.best_params_)

    final_model = fit_model(clone(search.best_estimator_), winner, X_train, y_train)
    proba, holdout = evaluate(final_model, X_hold, y_hold, config.RISK_THRESHOLD)

    metrics = {
        "best_model": winner,
        "selection": {
            "cv_folds": config.CV_FOLDS,
            "search_sample_frac": config.SEARCH_SAMPLE_FRAC,
            "candidate_cv": comparison,
            "best_params": search.best_params_,
            "best_grid_cv_auc": float(search.best_score_),
        },
        "holdout": holdout,
        "data": {
            "rows": int(len(df)),
            "features": int(X.shape[1]),
            "train_rows": int(len(X_train)),
            "holdout_rows": int(len(X_hold)),
            "target_rate": float(y.mean()),
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    # -------------------------------------------------------------------------
    # Persistência explícita dos dois artefatos usados em produção.
    #
    # abt_artifacts.pkl guarda o contrato de transformação da ABT: schema de
    # entrada e o preprocessor já AJUSTADO no conjunto de treino.
    #
    # model.pkl guarda somente o estimador vencedor já AJUSTADO e o threshold.
    #
    # Dessa forma, na inferência o fluxo é literalmente:
    # dados -> abt_artifacts.pkl -> model.pkl -> probabilidade.
    # -------------------------------------------------------------------------
    fitted_preprocessor = final_model.named_steps["preprocessor"]
    fitted_estimator = final_model.named_steps["model"]

    abt_artifacts = {
        "preprocessor": fitted_preprocessor,
        "feature_columns": X.columns.tolist(),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "transformed_feature_names": transformed_feature_names(final_model),
        "derived_features": [
            "CREDIT_INCOME_RATIO",
            "ANNUITY_INCOME_RATIO",
            "ANNUITY_CREDIT_RATIO",
        ],
        "history_prefixes": ["BUREAU_", "PREV_"],
        "target_column": config.TARGET_COL,
        "id_column": config.ID_COL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    model_bundle = {
        "model": fitted_estimator,
        "best_model": winner,
        "threshold": config.RISK_THRESHOLD,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    storage.write_pickle(abt_artifacts, config.ABT_ARTIFACTS_PATH)
    storage.write_pickle(model_bundle, config.MODEL_PATH)
    storage.write_json(metrics, config.METRICS_PATH)

    ids = df.loc[X_hold.index, config.ID_COL].values if config.ID_COL in df.columns else X_hold.index
    storage.write_csv(
        pd.DataFrame({config.ID_COL: ids, "TARGET": y_hold.values, "PD_DEFAULT": proba}),
        "reports/holdout_predictions.csv",
    )
    storage.write_csv(
        pd.DataFrame([confusion_metrics(y_hold, proba, t) for t in config.ANALYSIS_THRESHOLDS]),
        "reports/threshold_analysis.csv",
    )
    fpr, tpr, thresholds = roc_curve(y_hold, proba)
    storage.write_csv(
        pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds}),
        "reports/roc_curve_best_model.csv",
    )
    save_explainability(final_model, X_hold, y_hold)

    LOGGER.info(
        "Holdout: AUC=%.4f | KS=%.4f | Recall=%.4f",
        holdout["auc_roc"],
        holdout["ks"],
        holdout["recall_default"],
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


if __name__ == "__main__":
    train()
