"""Data engineering: raw CSV  ->  validated + cleaned + labelled  ->  processed v2."""
import json
import unicodedata

import pandas as pd

import tier_logic  # root-level module (also the CI Practical-1 program)
from .config import PROCESSED_PATH, RAW_PATH, REPORTS_DIR
from .versioning import record_version

EXPECTED_COLUMNS = ["team", "team_code", "association", "rank", "previous_rank", "points", "previous_points"]
KNOWN_ASSOCIATIONS = {"AFC", "CAF", "CONCACAF", "CONMEBOL", "OFC", "UEFA"}


def load_raw(path=RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Raw data is missing columns: {missing}")


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean the raw table and return (clean_df, quality_report)."""
    report = {"rows_in": int(len(df)), "nulls_per_column": df.isna().sum().to_dict()}
    df = df[EXPECTED_COLUMNS].copy()

    # text columns: strip + unicode-normalise (Türkiye, Côte d'Ivoire, Curaçao ...)
    for col in ["team", "team_code", "association"]:
        df[col] = df[col].astype(str).map(lambda s: unicodedata.normalize("NFC", s.strip()))
    df["association"] = df["association"].str.upper()
    df["team_code"] = df["team_code"].str.upper()

    # numeric columns: enforce dtypes
    for col in ["rank", "previous_rank", "points", "previous_points"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna()                                     # missing values
    report["rows_dropped_missing"] = before - len(df)
    before = len(df)
    df = df.drop_duplicates().drop_duplicates(subset="team_code")   # duplicates
    report["rows_dropped_duplicates"] = before - len(df)

    df["rank"] = df["rank"].astype(int)
    df["previous_rank"] = df["previous_rank"].astype(int)

    # business-rule validation
    bad_assoc = sorted(set(df["association"]) - KNOWN_ASSOCIATIONS)
    if bad_assoc:
        raise ValueError(f"Unknown associations: {bad_assoc}")
    if (df[["points", "previous_points"]] <= 0).any().any():
        raise ValueError("Points must be positive")
    if (df[["rank", "previous_rank"]] < 1).any().any():
        raise ValueError("Ranks must be >= 1")
    report["rank_is_unique"] = bool(df["rank"].is_unique)

    # informational IQR outlier count (points are bounded, so we keep them)
    q1, q3 = df["points"].quantile([0.25, 0.75])
    iqr = q3 - q1
    report["points_iqr_outliers"] = int(((df["points"] < q1 - 1.5 * iqr) | (df["points"] > q3 + 1.5 * iqr)).sum())

    df = df.sort_values("rank").reset_index(drop=True)
    report["rows_out"] = int(len(df))
    return df, report


def add_label_and_analysis_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tier"] = df["rank"].map(tier_logic.rank_to_tier)          # TARGET
    # analysis-only columns derived from CURRENT data -> never used as model features
    df["rank_change"] = df["previous_rank"] - df["rank"]
    df["points_change"] = (df["points"] - df["previous_points"]).round(2)
    return df


def build_processed(save: bool = True) -> pd.DataFrame:
    """Raw v1.0  ->  Processed v2.0."""
    raw = load_raw()
    validate_schema(raw)
    clean_df, report = clean(raw)
    processed = add_label_and_analysis_columns(clean_df)
    report["tier_counts"] = processed["tier"].value_counts().to_dict()
    if save:
        PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        processed.to_csv(PROCESSED_PATH, index=False)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "data_quality.json").write_text(json.dumps(report, indent=2))
        from .config import RAW_PATH as RP
        record_version("v1.0", "raw", RP, raw, None, "Original FIFA ranking CSV (2022-10-06), untouched.")
        record_version("v2.0", "processed", PROCESSED_PATH, processed, "v1.0",
                       "Validated, cleaned, unicode-normalised; tier label + analysis columns added.")
    return processed


if __name__ == "__main__":
    df = build_processed()
    print(df.head())
    print("Processed shape:", df.shape)
