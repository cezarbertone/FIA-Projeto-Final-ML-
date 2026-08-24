# Artefatos gerados pelo treinamento

Após executar `Model/train.py`, são criados:

- `DataPipeline/abt_artifacts.pkl`: contrato da ABT para inferência. Contém o `preprocessor` ajustado no treino, ordem das features de entrada, listas de features numéricas/categóricas, nomes das features transformadas e metadados do schema.
- `Model/model.pkl`: estimador vencedor já ajustado + threshold de decisão. O preprocessing não fica duplicado aqui.
- `Model/metrics.json`: métricas de seleção e holdout.
- `reports/model_comparison.csv`: CV-AUC dos candidatos.
- `reports/grid_search_results.csv`: resultados do GridSearchCV.
- `reports/holdout_predictions.csv`: TARGET e PD no holdout.
- `reports/threshold_analysis.csv`: impacto de thresholds 0.30, 0.50 e 0.70.
- `reports/roc_curve_best_model.csv`: pontos da curva ROC.
- `reports/feature_importance.csv`: importância nativa quando disponível.
- `reports/permutation_importance.csv`: importância por permutação.
- `reports/shap_importance.csv`: importância SHAP quando executável.

## Como a inferência usa os dois PKLs

```text
Entrada do Streamlit
      ↓
Model/predict.py
      ↓
DataPipeline/abt_artifacts.pkl
  - alinha schema
  - imputa valores
  - codifica categorias
  - aplica scaling quando necessário
      ↓
Model/model.pkl
  - calcula predict_proba
      ↓
PD + decisão
```

Assim, `abt_artifacts.pkl` e `model.pkl` são complementares e ambos são obrigatórios para scoring.
