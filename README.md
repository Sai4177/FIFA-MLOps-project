# FIFA MLOps — one project, all lab experiments

Uses **your dataset** `fifa_ranking_2022-10-06.csv` in every experiment.
**ML problem:** classify a national team's current FIFA tier (Elite / Strong / Mid / Developing) from its previous
ranking snapshot — `previous_rank`, `previous_points`, `association` (no target leakage). Step-by-step commands: `docs/COMMANDS.md`.

| # | Lab experiment | Where it lives | Main command |
|---|---|---|---|
| 1 | Unit II assignment: data versions, 6 models, MLflow, Registry v1-v3 | `scripts/exp1_mlflow_registry.py`, `src/fifa_mlops/` | `python scripts/exp1_mlflow_registry.py` |
| 2 | Model deployment as API (MLflow + FastAPI) and Prefect | `api/fastapi_app.py`, `flows/prefect_pipeline.py` | `uvicorn api.fastapi_app:app` / `python flows/prefect_pipeline.py` |
| 3 | End-to-end ML pipeline | `ml_pipeline.py` | `python ml_pipeline.py` |
| 4 | Apache Airflow 3.3.2 (WSL2) | `airflow/dags/fifa_ml_pipeline.py`, `airflow/setup_airflow_wsl.sh` | `airflow dags test fifa_ml_pipeline` |
| 5 | CI/CD Practicals 1-7 (GitHub Actions) | `.github/workflows/p1…p7`, `tier_logic.py`, `train_model.py`, `quality_gate.py`, `app.py`, `Dockerfile` | `bash scripts/run_cicd_local.sh` |

## Structure
```
data/raw|processed|features   dataset v1.0 → v2.0 → v3.0  (+ versions.json with SHA-256 lineage)
src/fifa_mlops/              config, data_engineering, feature_engineering, modeling, tracking (MLflow), versioning
scripts/                     prepare_data, exp1_mlflow_registry, preview_results, run_cicd_local.sh
api/  flows/  airflow/       Experiments 2a, 2b, 4
ml_pipeline.py               Experiment 3
tier_logic.py  train_model.py  quality_gate.py  app.py  test_*.py  Dockerfile  requirements-cicd.txt   Experiment 5
.github/workflows/           p1-first-ci … p7-ml-cicd-docker
tests/  docs/  reports/      extra tests, command guide, generated reports
```

## Verification status (what was actually executed when this project was built)
The build sandbox could not reach PyPI, so MLflow / Prefect / FastAPI / Airflow / Docker could not be installed or run there.

| Part | Status |
|---|---|
| Data engineering, versions v1→v3, feature engineering | **Run + unit-tested** |
| 6 models, evaluation, v1/v2/v3 hyper-parameter sets (`preview_results.py`) | **Run** (numbers in `reports/`) |
| `ml_pipeline.py` (Exp 3) | **Run** |
| Flask app, quality gate, ML tests, P1-P7 steps incl. failure demos (`run_cicd_local.sh`) | **Run, 15/15 checks as expected** |
| 7 workflow YAML files | **Linted with actionlint** (not executed on GitHub) |
| MLflow tracking/registry code (`tracking.py`) | Control-flow smoke-tested with an in-memory MLflow stub; **real MLflow run pending on your machine** |
| FastAPI app, Prefect flows, Airflow DAG | Written, byte-compiled; **not executed here** |
| Docker build/run (`Dockerfile`, P6/P7 container steps) | **Not executed here** (no Docker daemon); the same app was run from the release-candidate folder instead |
