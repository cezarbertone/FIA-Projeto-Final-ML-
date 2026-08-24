# Checklist de avaliação — Projeto Final

Este arquivo serve como mapa rápido para a defesa da solução implementada no repositório.

## Etapa individual

| Exigência | Evidência no projeto | Status |
|---|---|---|
| Arquitetura funcional completa da origem ao deploy | `README.md` seção 2; `Docs/architecture.md`; `MLOps/README.md` | OK |
| Infraestrutura com Docker Compose | `MLOps/docker-compose.yml` | OK |
| Serviço de predição | `app/app.py` + `Model/predict.py` | OK |
| Contrato explícito da ABT para inferência | `DataPipeline/abt_artifacts.pkl` gerado por `Model/train.py` | OK |
| Modelo persistido | `Model/model.pkl` gerado por `Model/train.py` | OK |
| Ações automatizadas a partir da previsão | `README.md` seção 13; `MLOps/README.md` seção 5 | OK |
| Agentes de IA em contexto aplicado | `MLOps/README.md` seção 6 | OK |

## Fundamentals

| Critério | Evidência |
|---|---|
| Fundamentação teórica | `README.md` seção 7 |
| Mecanismos técnicos | `README.md` seção 8 |
| Algoritmos | Logistic Regression, Random Forest e Gradient Boosting |
| Controle de overfitting | holdout intocado + CV + tuning separado |
| Desbalanceamento | class weights / sample weights |

## Coding

| Critério | Evidência |
|---|---|
| Qualidade de código | `DataPipeline/pipeline_functions.py`, `DataPipeline/config.py` e scripts enxutos por responsabilidade |
| Tratamento de dados | `DataPipeline/data_sanitization.py` + funções reutilizáveis em `DataPipeline/pipeline_functions.py` |
| Construção de pipeline | `DataPipeline/`, `pipeline_orchestration.py`, DAG Airflow |
| Configuração | `DataPipeline/config.py`, `Model/config.py`, variáveis Compose |
| Inferência | `Model/predict.py` |

## Estrutura do GitHub

### Dados

- `Dados/raw_data.csv` — gerado
- `Dados/clean_data.csv` — gerado
- `Dados/abt.csv` — gerado

### DataPipeline

- `pipeline_functions.py`
- `data_sanitization.py`
- `abt_transform.py`
- `exp_analysis.ipynb`
- `config.py`
- `abt_artifacts.pkl` — gerado no treinamento

### Model

- `train.py`
- `config.py`
- `evaluation.ipynb`
- `predict.py`
- `model.pkl` — gerado no treinamento
- `metrics.json` — gerado no treinamento

### MLOps

- `README.md`
- `docker-compose.yml`
- `pipeline_orchestration.py`
- `storage.py`

### App

- `app/app.py`

### Raiz

- `requirements.txt`
- `README.md`

## Checklist antes da banca

- [ ] três CSVs inseridos na Landing
- [ ] `docker compose ... up -d --build` executa sem erro
- [ ] Airflow abre em `localhost:8080`
- [ ] DAG executa até `06_score_sample`
- [ ] `Dados/raw_data.csv` existe
- [ ] `Dados/clean_data.csv` existe
- [ ] `Dados/abt.csv` existe
- [ ] `DataPipeline/abt_artifacts.pkl` existe
- [ ] `Model/model.pkl` existe
- [ ] `Model/metrics.json` existe
- [ ] Streamlit abre em `localhost:8501`
- [ ] tela mostra modelo, AUC, KS e Recall
- [ ] simulação de cliente retorna PD e decisão
- [ ] saber explicar de onde o Streamlit lê `abt.csv`, `metrics.json`, `abt_artifacts.pkl` e `model.pkl`
- [ ] explicabilidade possui ao menos uma saída disponível
- [ ] saber explicar as automações propostas e o papel do agente de IA
