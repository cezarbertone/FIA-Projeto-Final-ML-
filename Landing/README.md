# Landing

Coloque manualmente nesta pasta os três arquivos CSV do Home Credit:

- `application_train.csv`
- `bureau.csv`
- `previous_application.csv`

Na raiz do projeto, suba o ambiente com:

```powershell
docker compose -f .\MLOps\docker-compose.yml up -d --build
```

Depois execute a DAG `home_credit_risk_v3_pipeline` no Airflow (`http://localhost:8080`) ou rode o pipeline inteiro dentro do container:

```powershell
docker compose -f .\MLOps\docker-compose.yml exec airflow python /opt/project/MLOps/pipeline_orchestration.py
```

Nenhuma instalação de Python ou `.venv` é necessária no computador host.
