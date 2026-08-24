from __future__ import annotations

import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from DataPipeline.config import ABT_PATH, ID_COL, TARGET_COL, RISK_THRESHOLD, MINIO_BUCKET
from MLOps import storage
from Model.predict import score_dataframe

st.set_page_config(
    page_title="Home Credit — Risco de Inadimplência",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
/* =========================================================
   PALETA BASE
   ========================================================= */
:root {
    --hc-bg: #f6f8fc;
    --hc-surface: #ffffff;
    --hc-text: #0f172a;
    --hc-muted: #475569;
    --hc-border: #e5e7eb;
    --hc-sidebar: #0f172a;
    --hc-accent: #ef4444;
}

html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: var(--hc-bg) !important;
    color: var(--hc-text) !important;
    color-scheme: light !important;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}

/* =========================================================
   REGRA PRINCIPAL
   O Streamlit injeta cores em elementos internos dependendo
   do tema. Aqui forçamos os elementos textuais do MAIN.
   ========================================================= */
[data-testid="stMain"] p,
[data-testid="stMain"] span,
[data-testid="stMain"] label,
[data-testid="stMain"] small,
[data-testid="stMain"] li,
[data-testid="stMain"] a,
[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3,
[data-testid="stMain"] h4,
[data-testid="stMain"] h5,
[data-testid="stMain"] h6,
[data-testid="stMain"] [data-testid="stMarkdownContainer"],
[data-testid="stMain"] [data-testid="stMarkdownContainer"] *,
[data-testid="stMain"] [data-testid="stCaptionContainer"],
[data-testid="stMain"] [data-testid="stCaptionContainer"] *,
[data-testid="stMain"] [data-testid="stWidgetLabel"],
[data-testid="stMain"] [data-testid="stWidgetLabel"] *,
[data-testid="stMain"] [role="radiogroup"] label,
[data-testid="stMain"] [role="radiogroup"] label * {
    color: var(--hc-text) !important;
    -webkit-text-fill-color: var(--hc-text) !important;
}

/* =========================================================
   MÉTRICAS
   ========================================================= */
[data-testid="stMain"] [data-testid="stMetric"] {
    background: var(--hc-surface) !important;
    border: 1px solid var(--hc-border) !important;
    padding: .85rem 1rem !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 14px rgba(15,23,42,.04) !important;
}

[data-testid="stMain"] [data-testid="stMetricLabel"],
[data-testid="stMain"] [data-testid="stMetricLabel"] *,
[data-testid="stMain"] [data-testid="stMetricValue"],
[data-testid="stMain"] [data-testid="stMetricValue"] * {
    color: var(--hc-text) !important;
    -webkit-text-fill-color: var(--hc-text) !important;
}

[data-testid="stMain"] [data-testid="stMetricLabel"],
[data-testid="stMain"] [data-testid="stMetricLabel"] * {
    color: var(--hc-muted) !important;
    -webkit-text-fill-color: var(--hc-muted) !important;
}

/* =========================================================
   ABAS
   ========================================================= */
[data-testid="stMain"] [data-baseweb="tab-list"] button,
[data-testid="stMain"] [data-baseweb="tab-list"] button *,
[data-testid="stMain"] [role="tab"],
[data-testid="stMain"] [role="tab"] * {
    color: var(--hc-muted) !important;
    -webkit-text-fill-color: var(--hc-muted) !important;
}

[data-testid="stMain"] [data-baseweb="tab-list"] button[aria-selected="true"],
[data-testid="stMain"] [data-baseweb="tab-list"] button[aria-selected="true"] *,
[data-testid="stMain"] [role="tab"][aria-selected="true"],
[data-testid="stMain"] [role="tab"][aria-selected="true"] * {
    color: var(--hc-accent) !important;
    -webkit-text-fill-color: var(--hc-accent) !important;
}

/* =========================================================
   RADIO BUTTONS
   ========================================================= */
[data-testid="stMain"] [data-testid="stRadio"] > label,
[data-testid="stMain"] [data-testid="stRadio"] > label *,
[data-testid="stMain"] [data-testid="stRadio"] [role="radiogroup"] label,
[data-testid="stMain"] [data-testid="stRadio"] [role="radiogroup"] label * {
    color: var(--hc-text) !important;
    -webkit-text-fill-color: var(--hc-text) !important;
}

/* =========================================================
   INPUTS / SELECTS / NUMBER INPUT / FORM
   ========================================================= */
[data-testid="stMain"] input,
[data-testid="stMain"] textarea {
    color: var(--hc-text) !important;
    background-color: #ffffff !important;
    -webkit-text-fill-color: var(--hc-text) !important;
}

[data-testid="stMain"] [data-baseweb="input"],
[data-testid="stMain"] [data-baseweb="base-input"],
[data-testid="stMain"] [data-baseweb="select"] > div,
[data-testid="stMain"] [data-baseweb="select"] input {
    background-color: #ffffff !important;
    color: var(--hc-text) !important;
}

[data-testid="stMain"] [data-baseweb="select"] span,
[data-testid="stMain"] [data-baseweb="select"] div,
[data-testid="stMain"] [data-testid="stNumberInput"] span,
[data-testid="stMain"] [data-testid="stNumberInput"] button {
    color: var(--hc-text) !important;
    -webkit-text-fill-color: var(--hc-text) !important;
}

/* =========================================================
   ALERTAS: info / warning / success / error
   ========================================================= */
[data-testid="stMain"] [data-testid="stAlert"],
[data-testid="stMain"] [data-testid="stAlert"] div,
[data-testid="stMain"] [data-testid="stAlert"] p,
[data-testid="stMain"] [data-testid="stAlert"] span,
[data-testid="stMain"] [data-testid="stAlert"] * {
    color: var(--hc-text) !important;
    -webkit-text-fill-color: var(--hc-text) !important;
}

/* =========================================================
   EXPANDER / TABELAS / JSON / CODE
   ========================================================= */
[data-testid="stMain"] [data-testid="stExpander"] summary,
[data-testid="stMain"] [data-testid="stExpander"] summary *,
[data-testid="stMain"] [data-testid="stJson"] *,
[data-testid="stMain"] [data-testid="stDataFrame"] {
    color: var(--hc-text) !important;
    -webkit-text-fill-color: var(--hc-text) !important;
}

/* =========================================================
   SIDEBAR ESCURA
   ========================================================= */
[data-testid="stSidebar"] {
    background: var(--hc-sidebar) !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
    color: #f8fafc !important;
    -webkit-text-fill-color: #f8fafc !important;
}

/* Cards de métrica da sidebar */
[data-testid="stSidebar"] [data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid var(--hc-border) !important;
    padding: .85rem 1rem !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 14px rgba(15,23,42,.12) !important;
}

[data-testid="stSidebar"] [data-testid="stMetricLabel"],
[data-testid="stSidebar"] [data-testid="stMetricLabel"] * {
    color: var(--hc-muted) !important;
    -webkit-text-fill-color: var(--hc-muted) !important;
}

[data-testid="stSidebar"] [data-testid="stMetricValue"],
[data-testid="stSidebar"] [data-testid="stMetricValue"] * {
    color: var(--hc-text) !important;
    -webkit-text-fill-color: var(--hc-text) !important;
}

/* =========================================================
   COMPONENTES CUSTOMIZADOS
   ========================================================= */
.hero {
    padding: 1.5rem 1.7rem;
    border-radius: 18px;
    background: linear-gradient(120deg,#0f172a 0%,#1e3a8a 55%,#2563eb 100%);
    box-shadow: 0 12px 35px rgba(30,58,138,.20);
    margin-bottom: 1rem;
}

.hero,
.hero * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

.hero h1 {
    margin: 0 0 .35rem;
    font-size: 2rem;
}

.hero p {
    margin: 0;
    color: #e2e8f0 !important;
    -webkit-text-fill-color: #e2e8f0 !important;
}

.soft-card {
    background: #ffffff;
    border: 1px solid var(--hc-border);
    border-radius: 14px;
    padding: 1rem 1.15rem;
    box-shadow: 0 5px 16px rgba(15,23,42,.05);
    min-height: 110px;
}

.soft-label {
    font-size: .76rem;
    color: var(--hc-muted) !important;
    -webkit-text-fill-color: var(--hc-muted) !important;
    text-transform: uppercase;
    letter-spacing: .05em;
    font-weight: 700;
}

.soft-value {
    font-size: 1.3rem;
    font-weight: 750;
    color: var(--hc-text) !important;
    -webkit-text-fill-color: var(--hc-text) !important;
    margin-top: .25rem;
}

.soft-note {
    font-size: .78rem;
    color: #64748b !important;
    -webkit-text-fill-color: #64748b !important;
    margin-top: .2rem;
}

.decision-approved,
.decision-denied {
    padding: 1.25rem 1.4rem;
    border-radius: 16px;
    margin: .4rem 0 1rem;
    box-shadow: 0 8px 22px rgba(15,23,42,.10);
}

.decision-approved {
    background: #ecfdf5;
    border: 1px solid #86efac;
}

.decision-approved,
.decision-approved * {
    color: #14532d !important;
    -webkit-text-fill-color: #14532d !important;
}

.decision-denied {
    background: #fef2f2;
    border: 1px solid #fca5a5;
}

.decision-denied,
.decision-denied * {
    color: #7f1d1d !important;
    -webkit-text-fill-color: #7f1d1d !important;
}

.decision-title {
    font-size: 1.35rem;
    font-weight: 800;
    margin-bottom: .3rem;
}

/* =========================================================
   BOTÕES
   ========================================================= */
[data-testid="stMain"] button[kind="primary"],
[data-testid="stMain"] button[kind="primary"] *,
[data-testid="stMain"] [data-testid="stFormSubmitButton"] button,
[data-testid="stMain"] [data-testid="stFormSubmitButton"] button * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* Botões secundários permanecem legíveis sobre fundo claro */
[data-testid="stMain"] button[kind="secondary"],
[data-testid="stMain"] button[kind="secondary"] * {
    color: var(--hc-text) !important;
    -webkit-text-fill-color: var(--hc-text) !important;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=30, show_spinner=False)
def load_csv(key: str):
    return storage.read_csv(key, low_memory=False) if storage.exists(key) else None


@st.cache_data(ttl=30, show_spinner=False)
def load_json(key: str):
    return storage.read_json(key) if storage.exists(key) else None


def fmt_pct(value):
    return "-" if value is None else f"{float(value):.2%}"


def fmt_num(value, digits=4):
    return "-" if value is None else f"{float(value):.{digits}f}"


def money(value):
    if value is None or pd.isna(value):
        return "-"
    return f"R$ {float(value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def card(label, value, note=""):
    st.markdown(
        f'<div class="soft-card"><div class="soft-label">{label}</div><div class="soft-value">{value}</div><div class="soft-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def gauge(pd_value: float, threshold: float):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pd_value,
            number={"valueformat": ".3f", "font": {"size": 46, "color": "#0f172a"}},
            title={"text": "Probabilidade de inadimplência", "font": {"size": 18, "color": "#0f172a"}},
            gauge={
                "axis": {"range": [0, 1], "tickformat": ".1f"},
                "bar": {"color": "#2563eb"},
                "bgcolor": "white",
                "steps": [
                    {"range": [0, threshold], "color": "#dcfce7"},
                    {"range": [threshold, 1], "color": "#fee2e2"},
                ],
                "threshold": {"line": {"color": "#111827", "width": 5}, "thickness": .82, "value": threshold},
            },
        )
    )
    fig.update_layout(height=360, margin=dict(l=35, r=35, t=65, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def show_decision(row: pd.DataFrame, threshold: float, source_label: str):
    scored = score_dataframe(row, threshold=threshold)
    probability = float(scored["PD_DEFAULT"].iloc[0])
    decision = str(scored["CREDIT_DECISION"].iloc[0])
    suggestion = str(scored["ACTION_SUGGESTION"].iloc[0])
    approved = decision == "APROVAR"

    left, right = st.columns([1.25, 1])
    with left:
        st.plotly_chart(gauge(probability, threshold), use_container_width=True)
    with right:
        css = "decision-approved" if approved else "decision-denied"
        icon = "✅" if approved else "⛔"
        title = "APROVAR" if approved else "NEGAR / REVISAR"
        client_id = row[ID_COL].iloc[0] if ID_COL in row.columns else "simulação"
        st.markdown(
            f'<div class="{css}"><div class="decision-title">{icon} {title}</div>'
            f'<div><b>Origem:</b> {source_label}</div><div><b>Solicitação:</b> {client_id}</div>'
            f'<div><b>PD:</b> {probability:.3f}</div><div><b>Threshold:</b> {threshold:.2f}</div></div>',
            unsafe_allow_html=True,
        )
        st.info(suggestion)


metrics = load_json("Model/metrics.json")
abt = load_csv(ABT_PATH)

with st.sidebar:
    st.title("⚙️ Política de crédito")
    threshold = st.slider("Threshold de decisão", 0.00, 1.00, float(RISK_THRESHOLD), 0.01, format="%.2f")
    st.caption("PD abaixo do threshold → APROVAR. PD igual ou acima → NEGAR / REVISAR.")
    st.divider()
    if metrics:
        hold = metrics.get("holdout", {})
        st.markdown("### Modelo atual")
        st.write(f"**{metrics.get('best_model', '-').replace('_', ' ').title()}**")
        st.metric("AUC-ROC", fmt_num(hold.get("auc_roc")))
        st.metric("KS", fmt_num(hold.get("ks")))
    if st.button("Atualizar tela", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.markdown(
    '<div class="hero"><h1>💳 Home Credit — Score de Risco</h1><p>Scoring de crédito, métricas do modelo, política de threshold e explicabilidade.</p></div>',
    unsafe_allow_html=True,
)

hold = metrics.get("holdout", {}) if metrics else {}
c1, c2, c3, c4 = st.columns(4)
c1.metric("Modelo vencedor", metrics.get("best_model", "-").replace("_", " ").title() if metrics else "-")
c2.metric("AUC-ROC", fmt_num(hold.get("auc_roc")))
c3.metric("KS", fmt_num(hold.get("ks")))
c4.metric("Recall inadimplente", fmt_pct(hold.get("recall_default")))

tabs = st.tabs([
    "🎯 Decisão de crédito",
    "🧪 Treinamento e resultados",
    "🧮 Matriz de confusão",
    "🔎 Explicabilidade",
    "🪣 Data lake",
])

with tabs[0]:
    if abt is None or not storage.exists("Model/model.pkl") or not storage.exists("DataPipeline/abt_artifacts.pkl"):
        st.warning("Execute o pipeline até o final para gerar a ABT, abt_artifacts.pkl e model.pkl.")
    else:
        mode = st.radio(
            "Modo da simulação",
            ["Cliente histórico aleatório", "Nova solicitação simplificada"],
            horizontal=True,
        )
        if mode == "Cliente histórico aleatório":
            if st.button("🎲 Gerar cliente aleatório", type="primary"):
                st.session_state["row_idx"] = secrets.randbelow(len(abt))
            if "row_idx" not in st.session_state:
                st.info("Clique em **Gerar cliente aleatório**.")
            else:
                row = abt.iloc[[min(st.session_state["row_idx"], len(abt) - 1)]].copy()
                show_decision(row, threshold, "ABT histórica")
                st.markdown("### Variáveis do cliente")
                specs = [
                    ("Renda total", "AMT_INCOME_TOTAL", "money"),
                    ("Crédito solicitado", "AMT_CREDIT", "money"),
                    ("Anuidade", "AMT_ANNUITY", "money"),
                    ("Score externo 2", "EXT_SOURCE_2", "num"),
                    ("Créditos no bureau", "BUREAU_CREDIT_COUNT", "num"),
                    ("Dívida bureau", "BUREAU_AMT_DEBT_TOTAL", "money"),
                    ("Maior atraso bureau", "BUREAU_DAY_OVERDUE_MAX", "num"),
                    ("Taxa aprovação anterior", "PREV_APPROVAL_RATE", "pct"),
                ]
                visible = [item for item in specs if item[1] in row.columns]
                for start in range(0, len(visible), 4):
                    columns = st.columns(4)
                    for col, (label, key, kind) in zip(columns, visible[start:start + 4]):
                        with col:
                            value = row[key].iloc[0]
                            rendered = money(value) if kind == "money" else (fmt_pct(value) if kind == "pct" else fmt_num(value, 2))
                            card(label, rendered, key)
                with st.expander("Registro completo"):
                    st.dataframe(row.T, use_container_width=True, height=500)
        else:
            st.caption("Campos ausentes são tratados pelo Pipeline. BUREAU_* e PREV_* entram como zero quando não há histórico disponível.")
            with st.form("new_request"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    income = st.number_input("Renda anual", 0.0, 10_000_000.0, 180_000.0, 1_000.0)
                    credit = st.number_input("Crédito solicitado", 0.0, 10_000_000.0, 600_000.0, 1_000.0)
                    annuity = st.number_input("Anuidade/parcela", 0.0, 2_000_000.0, 27_000.0, 500.0)
                with col2:
                    goods = st.number_input("Valor do bem", 0.0, 10_000_000.0, 540_000.0, 1_000.0)
                    age = st.number_input("Idade", 18, 100, 35)
                    years = st.number_input("Anos de emprego", 0.0, 70.0, 5.0, .5)
                with col3:
                    ext1 = st.slider("EXT_SOURCE_1", 0.0, 1.0, .51, .01)
                    ext2 = st.slider("EXT_SOURCE_2", 0.0, 1.0, .57, .01)
                    ext3 = st.slider("EXT_SOURCE_3", 0.0, 1.0, .54, .01)
                gender = st.selectbox("Gênero", ["F", "M"])
                education = st.selectbox(
                    "Escolaridade",
                    ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"],
                )
                income_type = st.selectbox(
                    "Tipo de renda",
                    ["Working", "Commercial associate", "Pensioner", "State servant", "Businessman"],
                )
                submitted = st.form_submit_button("Calcular score", type="primary", use_container_width=True)
            if submitted:
                row = pd.DataFrame([{
                    "SK_ID_CURR": 999999,
                    "AMT_INCOME_TOTAL": income,
                    "AMT_CREDIT": credit,
                    "AMT_ANNUITY": annuity,
                    "AMT_GOODS_PRICE": goods,
                    "DAYS_BIRTH": -age * 365,
                    "DAYS_EMPLOYED": -years * 365,
                    "EXT_SOURCE_1": ext1,
                    "EXT_SOURCE_2": ext2,
                    "EXT_SOURCE_3": ext3,
                    "CODE_GENDER": gender,
                    "NAME_EDUCATION_TYPE": education,
                    "NAME_INCOME_TYPE": income_type,
                }])
                show_decision(row, threshold, "nova solicitação")

with tabs[1]:
    st.subheader("Seleção e avaliação")
    comparison = load_csv("reports/model_comparison.csv")
    if comparison is not None:
        st.dataframe(comparison, use_container_width=True)
    if metrics:
        selection = metrics.get("selection", {})
        holdout = metrics.get("holdout", {})
        a, b, c, d = st.columns(4)
        a.metric("CV folds", selection.get("cv_folds", "-"))
        b.metric("Amostra da busca", fmt_pct(selection.get("search_sample_frac")))
        c.metric("CV-AUC ajustada", fmt_num(selection.get("best_grid_cv_auc")))
        d.metric("Holdout", f"{holdout.get('rows', 0):,}".replace(",", "."))
        st.markdown("#### Hiperparâmetros vencedores")
        st.json(selection.get("best_params", {}))
        roc = load_csv("reports/roc_curve_best_model.csv")
        if roc is not None:
            fig = px.line(roc, x="fpr", y="tpr", title="Curva ROC — holdout")
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="#64748b"))
            st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    holdout_predictions = load_csv("reports/holdout_predictions.csv")
    if holdout_predictions is None:
        st.info("Execute o treinamento para gerar o holdout.")
    else:
        y = holdout_predictions[TARGET_COL].astype(int)
        probability = holdout_predictions["PD_DEFAULT"].astype(float)
        pred = (probability >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        st.caption("Classe positiva = inadimplente. O falso negativo representa um inadimplente classificado abaixo do threshold.")
        a, b, c, d = st.columns(4)
        a.metric("VN", f"{tn:,}".replace(",", "."))
        b.metric("FP", f"{fp:,}".replace(",", "."))
        c.metric("FN", f"{fn:,}".replace(",", "."))
        d.metric("VP", f"{tp:,}".replace(",", "."))
        matrix = pd.DataFrame(
            [[tn, fp], [fn, tp]],
            index=["Real: adimplente", "Real: inadimplente"],
            columns=["Previsto: aprovar", "Previsto: risco"],
        )
        fig = px.imshow(matrix, text_auto=True, aspect="auto", title=f"Matriz de confusão — threshold {threshold:.2f}")
        st.plotly_chart(fig, use_container_width=True)
        x, yc, z = st.columns(3)
        x.metric("Taxa de aprovação", fmt_pct((pred == 0).mean()))
        yc.metric("Recall inadimplente", fmt_pct(recall_score(y, pred, zero_division=0)))
        z.metric("Precision inadimplente", fmt_pct(precision_score(y, pred, zero_division=0)))
        threshold_table = load_csv("reports/threshold_analysis.csv")
        if threshold_table is not None:
            st.markdown("#### Políticas predefinidas")
            st.dataframe(threshold_table, use_container_width=True)

with tabs[3]:
    st.subheader("O que mais influencia o modelo")
    native = load_csv("reports/feature_importance.csv")
    permutation = load_csv("reports/permutation_importance.csv")
    shap = load_csv("reports/shap_importance.csv")
    for title, frame, value_col in [
        ("Importância nativa", native, "abs_value"),
        ("Permutation importance", permutation, "importance_mean"),
        ("SHAP médio absoluto", shap, "mean_abs_shap"),
    ]:
        if frame is not None and value_col in frame.columns:
            top = frame.head(20).sort_values(value_col)
            fig = px.bar(top, x=value_col, y="feature", orientation="h", title=title)
            fig.update_layout(height=550)
            st.plotly_chart(fig, use_container_width=True)
    if native is None and permutation is None and shap is None:
        st.info("Execute o treinamento para gerar as análises.")

with tabs[4]:
    st.subheader("Objetos do projeto")
    st.markdown(f"**Bucket configurado:** `{MINIO_BUCKET}`")
    st.code(
        "Dados/raw_data.csv\nDados/clean_data.csv\nDados/abt.csv\nDados/_processing/...\nDataPipeline/abt_artifacts.pkl\nModel/model.pkl\nModel/metrics.json\nreports/...",
        language="text",
    )
    keys = storage.list_keys("") if storage.STORAGE_BACKEND == "minio" else []
    if keys:
        st.dataframe(pd.DataFrame({"object_key": keys}), use_container_width=True, height=400)
