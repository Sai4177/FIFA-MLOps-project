"""Step 1+2: build processed v2 and feature-engineered v3 datasets."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fifa_mlops.data_engineering import build_processed
from fifa_mlops.feature_engineering import build_features
from fifa_mlops.versioning import load_versions

if __name__ == "__main__":
    p = build_processed(); print("v2 processed:", p.shape)
    f = build_features(); print("v3 features :", f.shape)
    for d in load_versions()["datasets"]:
        print(f"  {d['version']:5} {d['stage']:20} {d['rows']}x{d['columns']}  {d['file']}  sha256={d['sha256'][:12]}")
