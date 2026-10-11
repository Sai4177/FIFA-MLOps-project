"""EXPERIMENT 1 (Unit II assignment): 6 models -> MLflow tracking -> Model Registry v1, v2, v3.

Run from project root:   python scripts/exp1_mlflow_registry.py
Then open the UI:        mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from fifa_mlops import tracking as T
from fifa_mlops.config import MLFLOW_TRACKING_URI, REPORTS_DIR

if __name__ == "__main__":
    print("MLflow tracking URI:", MLFLOW_TRACKING_URI)
    results = T.run_comparison()
    table = T.comparison_table(results)
    print("\n=== 6-MODEL COMPARISON (v1 baseline hyper-parameters) ===")
    print(table.drop(columns="run_id").to_string(index=False))
    reg = T.register_versions(results)
    T.save_summary(results, reg, REPORTS_DIR / "registry_summary.json")
    print("\nBest model            :", reg["best_model"])
    for v in reg["versions"]:
        print(f"Registered Version {v['version']} [{v['label']:8}] acc={v['metrics']['accuracy']:.4f} f1={v['metrics']['f1_macro']:.4f} cv={v['metrics']['cv_accuracy_mean']:.4f}")
    print("Champion alias ->     : Version", reg["champion_version"])
