# Convenience targets (Linux/WSL/macOS).  Windows users: copy the commands from docs/COMMANDS.md
PY ?= python

install:      ; $(PY) -m pip install -r requirements.txt
data:         ; $(PY) scripts/prepare_data.py                      # raw v1 -> processed v2 -> features v3
exp1:         ; $(PY) scripts/exp1_mlflow_registry.py             # 6 models + MLflow + registry v1,v2,v3
mlflow-ui:    ; mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
api:          ; uvicorn api.fastapi_app:app --reload --port 8000  # Exp 2a  (needs exp1 first)
prefect:      ; $(PY) flows/prefect_pipeline.py                   # Exp 2b
prefect-mon:  ; $(PY) flows/prefect_pipeline.py --monitor
pipeline:     ; $(PY) ml_pipeline.py                              # Exp 3
airflow-test: ; FIFA_MLOPS_HOME=$(CURDIR) airflow dags test fifa_ml_pipeline   # Exp 4 (inside Airflow venv)
cicd-local:   ; bash scripts/run_cicd_local.sh                    # Exp 5 (P1..P7 replay)
test:         ; $(PY) -m unittest discover -v && PYTHONPATH=src:. $(PY) -m unittest tests.test_data_features tests.test_tracking_flow -v
preview:      ; $(PY) scripts/preview_results.py
all:          data exp1 pipeline test cicd-local
clean:        ; rm -rf mlflow.db mlruns mlartifacts artifacts/* outputs/* release_candidate fifa_tier_model.pkl metrics.json fifa_dataset.csv
