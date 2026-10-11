"""MLflow tracking + Model Registry logic (Unit II assignment, tasks 5-9).

NOTE: requires `pip install mlflow` (>=2.14, works with MLflow 3.x).
"""
import inspect
import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from . import modeling as M
from .config import (CHAMPION_ALIAS, EXPERIMENT_NAME, FEATURES_PATH, MLFLOW_ARTIFACT_ROOT,
                     MLFLOW_TRACKING_URI, PROCESSED_PATH, RAW_PATH, REGISTERED_MODEL_NAME, TIER_ORDER)
from .versioning import load_versions

# Types that skops must be told to trust when saving/loading the sklearn pipeline.
# Includes your own transformer plus the KNN internals (KDTree / BallTree / distance metrics).
TRUSTED_TYPES = [
    "fifa_mlops.feature_engineering.FifaFeatureEngineer",
    "sklearn.metrics._dist_metrics.EuclideanDistance64",
    "sklearn.metrics._dist_metrics.EuclideanDistance",
    "sklearn.neighbors._kd_tree.KDTree",
    "sklearn.neighbors._kd_tree.KDTree64",
    "sklearn.neighbors._ball_tree.BallTree",
    "sklearn.neighbors._ball_tree.BallTree64",
]


def setup_mlflow() -> str:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=MLFLOW_ARTIFACT_ROOT)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return MLFLOW_TRACKING_URI


def _simple_params(est) -> dict:
    return {f"clf__{k}": v for k, v in est.get_params().items()
            if isinstance(v, (int, float, str, bool, type(None)))}


def _log_model(pipe, X_train, preds):
    signature = infer_signature(X_train, preds)

    # MLflow 3.x renamed `artifact_path` -> `name`; pick whichever this version supports.
    params = inspect.signature(mlflow.sklearn.log_model).parameters
    path_kwarg = {"name": "model"} if "name" in params else {"artifact_path": "model"}

    kwargs = dict(
        sk_model=pipe,
        signature=signature,
        input_example=X_train.head(5),
        **path_kwarg,
    )

    try:
        return mlflow.sklearn.log_model(skops_trusted_types=TRUSTED_TYPES, **kwargs)
    except MlflowException as exc:
        if "untrusted" not in str(exc).lower():
            raise
        # Safety net: a model family produced a type not in TRUSTED_TYPES.
        # Add the type named in the error to TRUSTED_TYPES to use the safer skops format.
        print(f"[warn] skops rejected some types ({exc}); falling back to cloudpickle.")
        return mlflow.sklearn.log_model(
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            **kwargs,
        )


def _confusion_png(cm_df, title, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm_df.values, cmap="Blues")
    ax.set_xticks(range(len(TIER_ORDER)), TIER_ORDER, rotation=30)
    ax.set_yticks(range(len(TIER_ORDER)), TIER_ORDER)
    for i in range(cm_df.shape[0]):
        for j in range(cm_df.shape[1]):
            ax.text(j, i, int(cm_df.values[i, j]), ha="center", va="center")
    ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def train_and_log(name, estimator, split, phase, run_name=None) -> dict:
    """Train one model, log params/metrics/artifacts/model to MLflow. Returns result dict."""
    X_train, X_test, y_train, y_test = split
    pipe = M.build_pipeline(estimator).fit(X_train, y_train)
    metrics = M.evaluate(pipe, X_train, y_train, X_test, y_test)
    with mlflow.start_run(run_name=run_name or name) as run:
        mlflow.set_tags({"model_family": name, "phase": phase, "problem": "FIFA tier classification",
                         "dataset.raw": "v1.0", "dataset.processed": "v2.0", "dataset.features": "v3.0"})
        mlflow.log_params({"model": name, "train_rows": len(X_train), "test_rows": len(X_test),
                           "test_size": 0.2, "random_state": M.SEED, **_simple_params(estimator)})
        mlflow.log_metrics(metrics)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _confusion_png(M.confusion(pipe, X_test, y_test), f"{name} - confusion matrix", tmp / "confusion_matrix.png")
            M.confusion(pipe, X_test, y_test).to_csv(tmp / "confusion_matrix.csv")
            mlflow.log_artifact(str(tmp / "confusion_matrix.png"))
            mlflow.log_artifact(str(tmp / "confusion_matrix.csv"))
        mlflow.log_dict(load_versions(), "dataset_versions.json")
        for path, ctx in ((RAW_PATH, "raw-v1.0"), (PROCESSED_PATH, "processed-v2.0"), (FEATURES_PATH, "features-v3.0")):
            try:
                import pandas as pd
                ds = mlflow.data.from_pandas(pd.read_csv(path), source=str(path), name=ctx)
                mlflow.log_input(ds, context=ctx)
            except Exception as exc:   # dataset logging is optional metadata
                print("dataset logging skipped:", exc)
        info = _log_model(pipe, X_train, pipe.predict(X_train))
        run_id = run.info.run_id
    return {"name": name, "run_id": run_id, "model_uri": info.model_uri, "pipeline": pipe,
            "metrics": metrics, "phase": phase, "params": _simple_params(estimator)}


def run_comparison() -> list[dict]:
    """Task 5-7: train 6 models, log to MLflow, compare, return results (best via select_best)."""
    setup_mlflow()
    split = M.split_data()
    results = [train_and_log(n, e, split, phase="comparison") for n, e in M.get_models().items()]
    return results


def register_versions(results: list[dict]) -> dict:
    """Task 8-9: register best model as Version 1, retrain with new hyper-parameters
    -> Version 2, Version 3 ; alias `champion` -> best version."""
    setup_mlflow()
    client = MlflowClient()
    best = M.select_best(results)
    name = best["name"]
    split = M.split_data()

    versions = []
    mv = mlflow.register_model(best["model_uri"], REGISTERED_MODEL_NAME)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, mv.version, "description", "baseline hyper-parameters")
    versions.append({"version": int(mv.version), "label": "baseline", "params": best["params"], "metrics": best["metrics"]})

    for i, params in enumerate(M.TUNED_PARAMS[name], start=2):
        est = M.get_models()[name].set_params(**params)
        res = train_and_log(name, est, split, phase="tuning", run_name=f"{name}_tuned_v{i}")
        mv = mlflow.register_model(res["model_uri"], REGISTERED_MODEL_NAME)
        client.set_model_version_tag(REGISTERED_MODEL_NAME, mv.version, "description", f"tuned hyper-parameters set {i-1}: {params}")
        versions.append({"version": int(mv.version), "label": f"tuned-{i-1}", "params": res["params"], "metrics": res["metrics"]})

    champ = max(reversed(versions), key=lambda v: (round(v["metrics"]["f1_macro"], 6), v["metrics"]["cv_accuracy_mean"]))
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, str(champ["version"]))
    client.update_registered_model(REGISTERED_MODEL_NAME, description=f"FIFA tier classifier ({name}); champion = version {champ['version']}")
    return {"best_model": name, "versions": versions, "champion_version": champ["version"]}


def comparison_table(results: list[dict]):
    import pandas as pd
    rows = [{"model": r["name"], **{k: round(v, 4) for k, v in r["metrics"].items()}, "run_id": r["run_id"]} for r in results]
    return pd.DataFrame(rows).sort_values(["f1_macro", "cv_accuracy_mean"], ascending=False).reset_index(drop=True)


def save_summary(results, reg, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    comparison_table(results).to_csv(path.with_name("model_comparison.csv"), index=False)
    path.write_text(json.dumps(reg, indent=2, default=str))