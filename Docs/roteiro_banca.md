# Roteiro de apresentação / banca

## Estrutura dos 5 blocos técnicos

### 1. Problema de negócio

**Mensagem principal:** estimar a probabilidade de inadimplência para apoiar uma política de concessão de crédito.

Mostrar:

- problema;
- variável `TARGET`;
- saída `PD_DEFAULT`;
- diferença entre probabilidade do modelo e threshold da política.

### 2. Análise exploratória

Usar `DataPipeline/exp_analysis.ipynb`.

Mostrar:

- desbalanceamento do TARGET;
- valores ausentes;
- variáveis financeiras;
- `EXT_SOURCE_*`;
- cobertura de bureau e previous application.

### 3. ABT

Usar `DataPipeline/abt_overview.ipynb`.

Explicar:

- uma linha por `SK_ID_CURR`;
- até 15 features `BUREAU_*`;
- até 13 features `PREV_*`;
- três razões financeiras;
- por que as tabelas auxiliares são agregadas antes do join.

### 4. Modelo e ciclo de desenvolvimento

Mostrar o fluxo:

```text
80% treino / 20% holdout
        ↓
20% do treino para seleção/tuning
        ↓
3-fold CV
        ↓
Logistic / Random Forest / Gradient Boosting
        ↓
vencedor
        ↓
GridSearchCV
        ↓
fit final no treino completo
        ↓
holdout
```

Explicar:

- preprocessing dentro do Pipeline;
- imputação;
- encoding;
- desbalanceamento;
- como o holdout reduz risco de avaliação otimista;
- papel dos hiperparâmetros;
- separação final entre `abt_artifacts.pkl` e `model.pkl`.

### 5. Avaliação e explicabilidade

Usar `Model/evaluation.ipynb` e Streamlit.

Mostrar:

- AUC-ROC;
- KS;
- Recall;
- Precision;
- Matriz de confusão;
- threshold;
- feature importance / permutation importance / SHAP.

---

# Demonstração individual da arquitetura

Ordem recomendada:

1. mostrar `Docs/architecture.md`;
2. mostrar `MLOps/docker-compose.yml`;
3. abrir Airflow e mostrar a DAG;
4. mostrar os objetos no MinIO;
5. explicar que o Streamlit não lê a Landing diretamente;
6. mostrar `Dados/abt.csv` como fonte do modo histórico;
7. mostrar `DataPipeline/abt_artifacts.pkl` + `Model/model.pkl` como cadeia de inferência;
8. abrir Streamlit e realizar uma predição;
9. mostrar `Model/predict.py` como camada central de inferência;
10. finalizar com as automações e o agente de IA proposto.

## Resposta-chave: de onde o Streamlit puxa os dados?

> No ambiente Docker, o Streamlit usa `MLOps/storage.py` para acessar o MinIO. No modo de cliente histórico, ele lê `Dados/abt.csv`. As métricas vêm de `Model/metrics.json` e dos arquivos de `reports/`. Para calcular uma previsão, o Streamlit chama `Model/predict.py`, que carrega primeiro `DataPipeline/abt_artifacts.pkl` para transformar e alinhar a entrada e depois `Model/model.pkl` para calcular a probabilidade de inadimplência. Na nova solicitação, os dados de entrada vêm do formulário, mas o mesmo fluxo pelos dois PKLs é utilizado.

## Frase-resumo da arquitetura

> Os dados entram manualmente pela Landing, são orquestrados pelo Airflow e persistidos no MinIO. O pipeline gera os dados limpos, agrega os históricos e constrói uma ABT única por cliente. O treinamento produz `abt_artifacts.pkl`, `model.pkl` e as métricas de avaliação. O Streamlit consome os artefatos pelo `storage.py` e delega toda inferência ao `predict.py`, que transforma a entrada com o `abt_artifacts.pkl` antes de enviar os dados ao `model.pkl`.
