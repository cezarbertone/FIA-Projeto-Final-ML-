#!/usr/bin/env bash
set -euo pipefail
docker compose -f ./MLOps/docker-compose.yml exec airflow python /opt/project/MLOps/pipeline_orchestration.py
