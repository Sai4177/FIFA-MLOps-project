# FIFA MLOps project — command guide (all experiments)

Dataset used everywhere: `data/raw/fifa_ranking_2022-10-06.csv` (211 national teams).
ML problem: predict a team's **current FIFA tier** (Elite = rank 1-20, Strong 21-60, Mid 61-120, Developing 121+)
from its **previous** ranking snapshot (`previous_rank`, `previous_points`, `association`). Current rank/points are used
only to create the label, never as features.

> Windows users: run everything in *Anaconda Prompt* / PowerShell exactly as shown, except the `source ...activate`
> line (use `.venv\Scripts\activate`). Airflow (Experiment 4) needs WSL2 + Ubuntu, as in your lab guide.

## 0. One-time setup
```bash
cd fifa-mlops
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Unit II assignment — data versions, 6 models, MLflow, Model Registry
```bash
python scripts/prepare_data.py            # Raw v1.0 -> Processed v2.0 -> Feature-engineered v3.0  (+ data/versions.json)
python scripts/exp1_mlflow_registry.py    # trains 6 models, logs to MLflow, registers Version 1, 2, 3, alias "champion"
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000     # open http://127.0.0.1:5000
```
Screenshots to take in the MLflow UI: Experiments → `FIFA_Tier_Classification` (runs table, tick all runs → *Compare*),
one run's page (params, metrics, artifacts), **Models** → `FIFA_Tier_Classifier` (versions 1, 2, 3 + `champion` alias).
To start from a clean registry (so the first registered model is again *Version 1*): delete `mlflow.db` and `mlartifacts/`.

Check the numbers without MLflow: `python scripts/preview_results.py`

## 2a. Model deployment as REST API (MLflow Model Registry + FastAPI)  — needs step 1 first
```bash
uvicorn api.fastapi_app:app --reload --port 8000
```
Open `http://127.0.0.1:8000/docs` (Swagger): `GET /health`, `GET /features`, `POST /predict` with
```json
{"previous_rank": 3, "previous_points": 1770.65, "association": "CONMEBOL"}
```
Command-line test:
```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
     -d '{"previous_rank":3,"previous_points":1770.65,"association":"CONMEBOL"}'
```

## 2b. Prefect orchestration
```bash
python flows/prefect_pipeline.py              # runs fifa_mlops_pipeline once (data -> features -> 6 models -> registry)
prefect server start                          # optional UI at http://127.0.0.1:4200  (run in a 2nd terminal first)
python flows/prefect_pipeline.py --monitor    # second flow: calls the running FastAPI /health + /predict on 10 real teams
python flows/prefect_pipeline.py --serve      # schedules the pipeline daily at 06:00
```

## 3. End-to-end ML pipeline (Module 2 Practice 2)
```bash
python ml_pipeline.py
```
Creates `artifacts/fifa_pipeline.joblib`, `outputs/metrics.json`, `outputs/pipeline_report.txt`, `outputs/sample_prediction.csv`.

## 4. Apache Airflow 3.3.2 on WSL2 (follows your lab guide)
```bash
# inside Ubuntu (WSL2)
bash airflow/setup_airflow_wsl.sh /mnt/c/path/to/fifa-mlops      # or copy the project into ~/ first (faster)
source ~/airflow/airflow_venv/bin/activate
export AIRFLOW_HOME=~/airflow FIFA_MLOPS_HOME=/path/to/fifa-mlops
airflow dags list | grep fifa_ml_pipeline
airflow dags test fifa_ml_pipeline            # quick command-line run, prints every task log
airflow standalone                            # UI http://localhost:8080 (password: cat ~/airflow/simple_auth_manager_passwords.json.generated)
```
In the UI: Dags → search `fifa_ml_pipeline` → Trigger once → Graph view → click tasks for logs.
Scheduling demo: change `schedule=None` to `schedule="*/5 * * * *"` in `~/airflow/dags/fifa_ml_pipeline.py`, then revert.

## 5. CI/CD — the 7 GitHub Actions practicals
```bash
git init && git add . && git commit -m "FIFA MLOps project"
git branch -M main
git remote add origin https://github.com/<you>/fifa-mlops.git && git push -u origin main
```
| Practical | Workflow file | Trigger | Failure demo (Actions → *Run workflow*) |
|---|---|---|---|
| P1 First CI | `p1-first-ci.yml` | push/PR (tier files) + manual | tick **inject_bug** |
| P2 ML CI | `p2-ml-ci.yml` | manual | – |
| P3 Quality gate | `p3-quality-gate.yml` | manual | set **min_accuracy = 0.99** |
| P4 Preserve artifacts | `p4-preserve-artifacts.yml` | manual | tick **break_artifact_path** |
| P5 Continuous delivery | `p5-continuous-delivery.yml` | manual | tick **break_api_test** |
| P6 Docker-ready release | `p6-docker-ready-release.yml` | manual | – (then build locally, below) |
| P7 CI/CD with Docker | `p7-ml-cicd-docker.yml` | **every push/PR** + manual | tick **wrong_health_port** |

P6 manual Docker steps (download the `fifa-tier-ml-release` artifact, unzip, open a terminal in that folder):
```bash
docker build -t fifa-tier-ml:1.0 .
docker images
docker run -d --name fifa-tier-ml-container -p 5000:5000 fifa-tier-ml:1.0
docker ps
curl http://localhost:5000/health
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"previous_rank":3,"previous_points":1770.65,"association":"CONMEBOL"}'
docker logs fifa-tier-ml-container
docker stop fifa-tier-ml-container && docker rm fifa-tier-ml-container
```
Replay every practical locally (success + failure demos, no GitHub needed): `bash scripts/run_cicd_local.sh`

## 6. Run all tests
```bash
python -m unittest discover -v                                           # root tests: tier logic, ML pipeline, Flask API
PYTHONPATH=src:. python -m unittest tests.test_data_features tests.test_tracking_flow -v   # Windows: set PYTHONPATH=src;.
```
(`train_model.py` must run first for `test_ml_pipeline` / `test_app`.)
