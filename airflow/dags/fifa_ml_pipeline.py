"""EXPERIMENT 4 - Apache Airflow 3.x DAG on the FIFA ranking dataset.

load_data -> preprocess_data -> engineer_features -> train_models -> evaluate_and_select -> quality_gate -> save_model

Install the DAG:  see airflow/setup_airflow_wsl.sh  (copies/symlinks this file into ~/airflow/dags)
The project folder is found through the FIFA_MLOPS_HOME environment variable.
Only pandas / scikit-learn / joblib are needed in the Airflow virtual environment (no MLflow).
"""
import os
import sys
from pathlib import Path

import pendulum
from airflow.sdk import dag, task

PROJECT = Path(os.environ.get("FIFA_MLOPS_HOME", Path(__file__).resolve().parents[2]))
MIN_ACCURACY = float(os.environ.get("FIFA_MIN_ACCURACY", "0.85"))


def _imports():
    """Make the project package importable inside the task process."""
    for p in (str(PROJECT / "src"), str(PROJECT)):
        if p not in sys.path:
            sys.path.insert(0, p)


@dag(
    dag_id="fifa_ml_pipeline",
    schedule=None,                       # manual; change to "*/5 * * * *" to demo scheduling
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["mlops", "fifa"],
    doc_md=__doc__,
)
def fifa_ml_pipeline():

    @task
    def load_data() -> dict:
        _imports()
        from fifa_mlops.data_engineering import load_raw, validate_schema
        print("STEP 1: Loading raw FIFA ranking data (dataset v1.0)")
        df = load_raw()
        validate_schema(df)
        print("rows, columns =", df.shape)
        return {"rows": int(df.shape[0]), "columns": int(df.shape[1])}

    @task
    def preprocess_data(info: dict) -> str:
        _imports()
        from fifa_mlops.data_engineering import build_processed
        print("STEP 2: Data engineering (raw v1.0 -> processed v2.0). Input:", info)
        df = build_processed()
        print(df["tier"].value_counts().to_string())
        return "data/processed/fifa_processed_v2.csv"

    @task
    def engineer_features(processed_path: str) -> str:
        _imports()
        from fifa_mlops.feature_engineering import build_features
        print("STEP 3: Feature engineering (processed v2.0 -> features v3.0) from", processed_path)
        table = build_features()
        print("feature table shape:", table.shape)
        return "data/features/fifa_features_v3.csv"

    @task
    def train_models(features_path: str) -> list:
        _imports()
        from fifa_mlops.modeling import train_all
        print("STEP 4: Training 6 models on", features_path)
        results = train_all()
        summary = []
        for r in results:
            m = r["metrics"]
            print(f"{r['name']:<20} accuracy={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f} cv={m['cv_accuracy_mean']:.4f}")
            summary.append({"name": r["name"], **m})
        return summary

    @task
    def evaluate_and_select(summary: list) -> dict:
        print("STEP 5: Selecting best model (highest test macro-F1, tie-break CV accuracy)")
        best = max(summary, key=lambda m: (round(m["f1_macro"], 6), m["cv_accuracy_mean"]))
        print("Best model:", best["name"], "| accuracy =", round(best["accuracy"], 4))
        return best

    @task
    def quality_gate(best: dict) -> dict:
        print(f"STEP 6: Quality gate  accuracy={best['accuracy']:.4f}  required>={MIN_ACCURACY}")
        if best["accuracy"] < MIN_ACCURACY:
            raise ValueError("QUALITY GATE FAILED - model below the required accuracy")
        print("QUALITY GATE PASSED")
        return best

    @task
    def save_model(best: dict) -> str:
        _imports()
        import json
        import joblib
        from fifa_mlops.modeling import TUNED_PARAMS, build_pipeline, get_models, split_data
        print("STEP 7: Re-fitting the winner and saving the pipeline")
        X_train, _, y_train, _ = split_data()
        pipe = build_pipeline(get_models()[best["name"]].set_params(**TUNED_PARAMS[best["name"]][1])).fit(X_train, y_train)
        out = PROJECT / "artifacts" / "airflow_best_model.joblib"
        out.parent.mkdir(exist_ok=True)
        joblib.dump(pipe, out)
        (PROJECT / "outputs").mkdir(exist_ok=True)
        (PROJECT / "outputs" / "airflow_best_model.json").write_text(json.dumps(best, indent=2))
        print("saved:", out)
        return str(out)

    info = load_data()
    processed = preprocess_data(info)
    features = engineer_features(processed)
    summary = train_models(features)
    best = evaluate_and_select(summary)
    passed = quality_gate(best)
    save_model(passed)


fifa_ml_pipeline()
