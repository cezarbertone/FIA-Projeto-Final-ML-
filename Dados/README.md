# Dados gerados pelo pipeline

Os principais arquivos são criados automaticamente:

- `raw_data.csv`: cópia promovida de `application_train.csv`.
- `clean_data.csv`: base principal limpa, ainda no nível de uma solicitação por cliente.
- `abt.csv`: Analytical Base Table pronta para o split de modelagem.
- `_processing/`: fontes auxiliares e agregações intermediárias de `bureau` e `previous_application`.

Não é necessário criar esses arquivos manualmente.
