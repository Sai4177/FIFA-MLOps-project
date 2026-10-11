"""EXPERIMENT 2b - Workflow orchestration with Prefect.

Flow 1  fifa_mlops_pipeline  : data engineering -> features -> 6 models (MLflow) -> registry -> champion
Flow 2  api_monitoring_flow  : calls the FastAPI service (/health + /predict) and checks the answers

Run once:                 python flows/prefect_pipeline.py
Prefect UI (optional):    prefect server start      (http://127.0.0.1:4200)
Schedule (daily 06:00):   python flows/prefect_pipeline.py --serve
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import pandas as pd
import requests
from prefect import flow, get_run_logger, task

from fifa_mlops import tracking as T
from fifa_mlops.config import PROCESSED_PATH, REPORTS_DIR
from fifa_mlops.data_engineering import build_processed
from fifa_mlops.feature_engineering import build_features
from fifa_mlops.modeling import select_best


@task(name="data-engineering (raw v1 -> processed v2)", retries=2, retry_delay_seconds=3)
def data_engineering_task() -> int:
    df = build_processed()
    get_run_logger().info("Processed dataset v2.0 rows=%s cols=%s", *df.shape)
    return len(df)


@task(name="feature-engineering (processed v2 -> features v3)")
def feature_engineering_task(_upstream: int) -> int:
    df = build_features()
    get_run_logger().info("Feature dataset v3.0 rows=%s cols=%s", *df.shape)
    return df.shape[1]


@task(name="train-6-models + MLflow tracking")
def train_task(_upstream: int):
    results = T.run_comparison()
    log = get_run_logger()
    for _, row in T.comparison_table(results).iterrows():
        log.info("%-18s acc=%.3f f1=%.3f cv=%.3f", row["model"], row["accuracy"], row["f1_macro"], row["cv_accuracy_mean"])
    return results


@task(name="register-versions + champion alias")
def register_task(results):
    reg = T.register_versions(results)
    T.save_summary(results, reg, REPORTS_DIR / "registry_summary.json")
    get_run_logger().info("Best=%s champion=Version %s", reg["best_model"], reg["champion_version"])
    return reg


@flow(name="fifa-mlops-pipeline", log_prints=True)
def fifa_mlops_pipeline():
    rows = data_engineering_task()
    cols = feature_engineering_task(rows)
    results = train_task(cols)
    reg = register_task(results)
    print("Pipeline finished. Champion version:", reg["champion_version"])
    return reg


@task(retries=3, retry_delay_seconds=2)
def call_health(base_url: str) -> dict:
    r = requests.get(f"{base_url}/health", timeout=10)
    r.raise_for_status()
    return r.json()


@task
def call_predict(base_url: str, row: dict) -> dict:
    payload = {k: row[k] for k in ("previous_rank", "previous_points", "association")}
    r = requests.post(f"{base_url}/predict", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


@flow(name="fifa-api-monitoring", log_prints=True)
def api_monitoring_flow(base_url: str = "http://127.0.0.1:8000", n_samples: int = 10):
    """Smoke-checks the running FastAPI service with real teams from the processed dataset."""
    print("health:", call_health(base_url))
    sample = pd.read_csv(PROCESSED_PATH).sample(n_samples, random_state=1)
    hits = 0
    for _, row in sample.iterrows():
        out = call_predict(base_url, row.to_dict())
        ok = out["prediction"] == row["tier"]
        hits += ok
        print(f"{row['team']:<22} actual={row['tier']:<11} predicted={out['prediction']:<11} {'OK' if ok else 'x'}")
    print(f"API agreement with true tier: {hits}/{n_samples}")
    return hits / n_samples


if __name__ == "__main__":
    if "--serve" in sys.argv:
        fifa_mlops_pipeline.serve(name="fifa-daily", cron="0 6 * * *")
    elif "--monitor" in sys.argv:
        api_monitoring_flow()
    else:
        fifa_mlops_pipeline()
