"""Dataset version registry (data/versions.json): Raw -> Processed -> Features."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT, VERSIONS_PATH


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_versions() -> dict:
    if VERSIONS_PATH.exists():
        return json.loads(VERSIONS_PATH.read_text())
    return {"datasets": []}


def record_version(version: str, stage: str, path: Path, df, parent: str | None, notes: str) -> dict:
    entry = {
        "version": version,
        "stage": stage,
        "file": Path(path).resolve().relative_to(ROOT).as_posix(),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "sha256": sha256_file(Path(path)),
        "parent": parent,
        "notes": notes,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    reg = load_versions()
    reg["datasets"] = [d for d in reg["datasets"] if d["version"] != version]
    reg["datasets"].append(entry)
    reg["datasets"].sort(key=lambda d: d["version"])
    VERSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    VERSIONS_PATH.write_text(json.dumps(reg, indent=2))
    return entry
