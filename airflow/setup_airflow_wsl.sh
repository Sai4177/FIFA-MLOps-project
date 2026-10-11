#!/usr/bin/env bash
# EXPERIMENT 4 - Apache Airflow 3.3.2 on Windows WSL2 / Ubuntu (follows the lab guide).
# Usage (inside Ubuntu):  bash airflow/setup_airflow_wsl.sh /path/to/fifa-mlops
set -euo pipefail
PROJECT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"

sudo apt update && sudo apt install -y python3-pip python3-venv
mkdir -p ~/airflow && cd ~/airflow
python3 -m venv airflow_venv
source airflow_venv/bin/activate
pip install --upgrade pip

AIRFLOW_VERSION=3.3.2
PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"
# libraries needed by the DAG tasks (same constraints keep versions compatible)
pip install pandas scikit-learn joblib matplotlib --constraint "${CONSTRAINT_URL}"

export AIRFLOW_HOME=~/airflow
export FIFA_MLOPS_HOME="$PROJECT"
mkdir -p ~/airflow/dags
cp "$PROJECT/airflow/dags/fifa_ml_pipeline.py" ~/airflow/dags/

airflow version
airflow config get-value core dags_folder
echo
echo "NEXT:  export AIRFLOW_HOME=~/airflow FIFA_MLOPS_HOME=$PROJECT && airflow standalone"
echo "UI  :  http://localhost:8080   user=admin   password: cat ~/airflow/simple_auth_manager_passwords.json.generated"
