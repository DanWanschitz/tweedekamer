"""
Week 8 Systematic Bias Audit

This script audits possible systematic bias risks in a Dutch Tweede Kamer
parliamentary decision making dataset.

It does not train a new final production model. Instead, it creates tables,
model comparisons, subgroup diagnostics, robustness checks, class performance
checks, bias hypothesis summaries, and a written summary that can be explained
in a student presentation.

Run from the repository root with:

    py -3.12 week8_bias_audit.py

Expected main dataset:

    parliamentary_notebooks_speeches/speeches_with_sentiment.csv

Outputs are written to:

    01_bias_audit_outputs
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
MIN_OUTCOME_GROUP_N = 50
MIN_PERFORMANCE_GROUP_N = 100


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def find_repo_root() -> Path:
    """
    Find the repository root by looking for the expected modelling dataset.
    This helps when the script is run from VS Code, Downloads, or PowerShell.
    """
    expected_relative_path = Path("parliamentary_notebooks_speeches") / "speeches_with_sentiment.csv"

    candidate_roots: List[Path] = []

    def add_candidate(path: Path) -> None:
        try:
            resolved = path.expanduser().resolve()
        except Exception:
            return
        if resolved not in candidate_roots:
            candidate_roots.append(resolved)

    cwd = Path.cwd()
    add_candidate(cwd)
    for parent in cwd.parents:
        add_candidate(parent)

    try:
        script_dir = Path(__file__).resolve().parent
        add_candidate(script_dir)
        for parent in script_dir.parents:
            add_candidate(parent)
    except NameError:
        pass

    home = Path.home()
    common_roots = [
        home,
        home / "Desktop",
        home / "desktop",
        home / "Documents",
        home / "Downloads",
        home / "downloads",
        home / "Desktop" / "tweedekamer",
        home / "desktop" / "tweedekamer",
        home / "Documents" / "tweedekamer",
    ]
    for root in common_roots:
        add_candidate(root)

    for candidate in candidate_roots:
        if (candidate / expected_relative_path).exists():
            return candidate

    return Path.cwd().resolve()


REPO_ROOT = find_repo_root()
DATA_DIR = REPO_ROOT / "parliamentary_notebooks_speeches"
MAIN_DATA_PATH = DATA_DIR / "speeches_with_sentiment.csv"
FEATURE_META_PATH = DATA_DIR / "feature_cols.json"
CLASS_META_PATH = DATA_DIR / "class_meta.json"
OUTPUT_DIR = REPO_ROOT / "01_bias_audit_outputs"


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

TARGET_COL = "label"

KEY_GROUP_COLUMNS = [
    "label",
    "motion_outcome_raw",
    "speaker_party_clean",
    "time_bin",
    "year",
    "is_voorzitter",
    "sentiment_label",
]

MISSINGNESS_AUDIT_COLUMNS = [
    "speaker_party",
    "speaker_party_clean",
    "sentiment_score",
    "sentiment_pos",
    "sentiment_neg",
    "speech_duration_seconds",
    "log_speech_duration",
    "log_text_length",
    "time_diff",
    "Aanvangstijd_y",
    "Eindtijd_y",
]

SUBGROUP_OUTCOME_COLUMNS = [
    "speaker_party_clean",
    "time_bin",
    "is_voorzitter",
    "year",
    "sentiment_label",
    "long_speech_flag",
]

SUBGROUP_PERFORMANCE_COLUMNS = [
    "motion_outcome_raw",
    "speaker_party_clean",
    "time_bin",
    "is_voorzitter",
    "sentiment_label",
    "long_speech_flag",
    "year",
]

MODEL_A_FEATURES = [
    "hour_sin",
    "hour_cos",
    "time_bin_ordinal",
    "is_morning",
    "is_early_afternoon",
    "is_late_afternoon",
    "is_evening",
    "is_night",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "years_since_2010",
]

MODEL_B_EXTRA_FEATURES = [
    "is_voorzitter",
    "log_speech_duration",
    "log_text_length",
    "long_speech_flag",
    "log_speaker_freq",
    "voorzitter_x_evening",
    "voorzitter_x_night",
    "duration_x_evening",
    "sentiment_score",
    "sentiment_pos",
    "sentiment_neg",
]

PARTY_FEATURES = [
    "party_CDA",
    "party_ChristenUnie",
    "party_D66",
    "party_GroenLinks",
    "party_Other",
    "party_PVV",
    "party_PvdA",
    "party_PvdD",
    "party_SGP",
    "party_SP",
    "party_VVD",
]

MODEL_B_FEATURES = MODEL_A_FEATURES + MODEL_B_EXTRA_FEATURES
MODEL_C_FEATURES = MODEL_B_FEATURES + PARTY_FEATURES

SENTIMENT_FEATURES = [
    "sentiment_score",
    "sentiment_pos",
    "sentiment_neg",
]


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def warn(message: str) -> None:
    print(f"WARNING: {message}")
    warnings.warn(message, stacklevel=2)


def safe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def save_csv(df: pd.DataFrame, filename: str) -> Path:
    ensure_output_dir()
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    print(f"Saved {safe_relative(path)}")
    return path


def read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        warn(f"File not found, skipping: {safe_relative(path)}")
        return None

    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        warn(f"Could not read {path}: {exc}")
        return None


def available_columns(df: pd.DataFrame, columns: Sequence[str], context: str) -> List[str]:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        warn(f"{context}: skipping unavailable columns: {missing}")
    return [col for col in columns if col in df.columns]


def check_required_columns(df: pd.DataFrame) -> None:
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Required target column is missing: {TARGET_COL}. "
            f"The audit cannot continue without `{TARGET_COL}`."
        )

    if df[TARGET_COL].isna().all():
        raise ValueError(f"Target column `{TARGET_COL}` is entirely missing.")

    optional_but_important = [
        "motion_outcome_raw",
        "speaker_party_clean",
        "time_bin",
        "year",
        "is_voorzitter",
        "sentiment_label",
    ]
    missing_optional = [col for col in optional_but_important if col not in df.columns]
    if missing_optional:
        warn(
            "Important context columns are missing. The audit will continue, "
            f"but some subgroup tables will be skipped: {missing_optional}"
        )


def format_rate(value: float) -> str:
    if pd.isna(value):
        return "not available"
    return f"{100 * float(value):.1f} percent"


def format_int(value: object) -> str:
    if pd.isna(value):
        return "not available"
    try:
        return f"{int(value):,}"
    except Exception:
        return str(value)


def metric_dict(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "roc_auc": np.nan,
    }

    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        except Exception as exc:
            warn(f"Could not calculate ROC AUC: {exc}")

    return metrics


# ---------------------------------------------------------------------------
# Section 1. Load data and basic checks
# ---------------------------------------------------------------------------

def load_main_dataset() -> pd.DataFrame:
    print("\n1. Load data and basic checks")
    print("=" * 80)

    if not MAIN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Main modelling dataset not found: {MAIN_DATA_PATH}. "
            "Run this script from the repository root or check the path."
        )

    df = pd.read_csv(MAIN_DATA_PATH, low_memory=False)
    check_required_columns(df)

    print(f"Repository root: {REPO_ROOT}")
    print(f"Loaded: {safe_relative(MAIN_DATA_PATH)}")
    print(f"Shape: {df.shape[0]:,} rows and {df.shape[1]:,} columns")

    print("\nTarget distribution:")
    print(df[TARGET_COL].value_counts(dropna=False).sort_index())

    print("\nAvailable columns:")
    print(list(df.columns))

    missingness = (
        df.isna()
        .sum()
        .rename("missing_count")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    missingness["missing_rate"] = missingness["missing_count"] / len(df)
    print("\nBasic missingness, top 20 columns by missing rate:")
    print(
        missingness.sort_values(["missing_rate", "missing_count"], ascending=False)
        .head(20)
        .to_string(index=False)
    )

    if FEATURE_META_PATH.exists():
        try:
            with open(FEATURE_META_PATH, "r", encoding="utf-8") as f:
                feature_meta = json.load(f)
            print(f"\nLoaded feature metadata from {safe_relative(FEATURE_META_PATH)}")
            print(f"Feature metadata keys: {list(feature_meta.keys())}")
        except Exception as exc:
            warn(f"Could not read feature metadata: {exc}")
    else:
        warn(f"Feature metadata not found: {safe_relative(FEATURE_META_PATH)}")

    if CLASS_META_PATH.exists():
        try:
            with open(CLASS_META_PATH, "r", encoding="utf-8") as f:
                class_meta = json.load(f)
            print(f"\nLoaded class metadata from {safe_relative(CLASS_META_PATH)}")
            print(class_meta)
        except Exception as exc:
            warn(f"Could not read class metadata: {exc}")
    else:
        warn(f"Class metadata not found: {safe_relative(CLASS_META_PATH)}")

    return df


# ---------------------------------------------------------------------------
# Section 2. Missingness audit
# ---------------------------------------------------------------------------

def missingness_overall(df: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_rate": df.isna().mean().values,
            "dtype": [str(dtype) for dtype in df.dtypes],
        }
    )
    table = table.sort_values(
        ["missing_rate", "missing_count", "column"],
        ascending=[False, False, True],
    )
    return table


def missingness_by_group(
    df: pd.DataFrame,
    group_col: str,
    audit_columns: Sequence[str],
) -> pd.DataFrame:
    output_columns = [
        "group_column",
        "group_value",
        "n",
        "audited_column",
        "missing_count",
        "missing_rate",
    ]

    if group_col not in df.columns:
        warn(f"Missingness audit: group column `{group_col}` is unavailable.")
        return pd.DataFrame(columns=output_columns)

    cols = available_columns(df, audit_columns, context=f"Missingness audit by `{group_col}`")
    if not cols:
        return pd.DataFrame(columns=output_columns)

    selected_cols = [group_col] + [col for col in cols if col != group_col]
    work = df[selected_cols].copy()
    work[group_col] = work[group_col].astype("object").where(work[group_col].notna(), "MISSING_GROUP_VALUE")

    rows = []
    for group_value, group_df in work.groupby(group_col, dropna=False):
        n = len(group_df)
        for col in cols:
            missing_count = int(group_df[col].isna().sum())
            rows.append(
                {
                    "group_column": group_col,
                    "group_value": str(group_value),
                    "n": n,
                    "audited_column": col,
                    "missing_count": missing_count,
                    "missing_rate": missing_count / n if n else np.nan,
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["group_column", "audited_column", "missing_rate", "group_value"],
        ascending=[True, True, False, True],
    )


def run_missingness_audit(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    print("\n2. Missingness audit")
    print("=" * 80)

    results: Dict[str, pd.DataFrame] = {}

    overall = missingness_overall(df)
    save_csv(overall, "missingness_overall.csv")
    results["overall"] = overall

    for group_col in KEY_GROUP_COLUMNS:
        table = missingness_by_group(df, group_col, MISSINGNESS_AUDIT_COLUMNS)
        save_csv(table, f"missingness_by_{group_col}.csv")
        results[group_col] = table

    print("\nPrinted interpretation of main missingness issues:")
    for col in ["Aanvangstijd_y", "Eindtijd_y", "speaker_party"]:
        if col in df.columns:
            rate = df[col].isna().mean()
            count = int(df[col].isna().sum())
            print(f"  {col}: {count:,} missing values, {format_rate(rate)} missing.")
        else:
            print(f"  {col}: column not available in this dataset.")

    aanvang_rate = df["Aanvangstijd_y"].isna().mean() if "Aanvangstijd_y" in df.columns else np.nan
    eind_rate = df["Eindtijd_y"].isna().mean() if "Eindtijd_y" in df.columns else np.nan
    party_rate = df["speaker_party"].isna().mean() if "speaker_party" in df.columns else np.nan

    print(
        "\nAudit interpretation: Aanvangstijd_y and Eindtijd_y should be treated "
        "as major missingness concerns. If they are fully missing, then time "
        "information from the original sources was not retained in those raw "
        "columns and any time based analysis depends on engineered replacements. "
        f"In this run their missing rates are {format_rate(aanvang_rate)} and "
        f"{format_rate(eind_rate)}."
    )
    print(
        "Audit interpretation: speaker_party missingness is important because "
        "party is a central political context variable. In this run the observed "
        f"missingness is {format_rate(party_rate)}. This directly connects data "
        "cleaning choices to data linkage and missingness bias."
    )

    return results


# ---------------------------------------------------------------------------
# Section 3. Linkage and pipeline loss audit
# ---------------------------------------------------------------------------

def audit_file(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {
            "file_name": safe_relative(path),
            "exists": False,
            "n_rows": np.nan,
            "n_columns": np.nan,
            "missing_cells": np.nan,
            "percent_missing_cells": np.nan,
        }

    df = read_csv_if_exists(path)
    if df is None:
        return {
            "file_name": safe_relative(path),
            "exists": True,
            "n_rows": np.nan,
            "n_columns": np.nan,
            "missing_cells": np.nan,
            "percent_missing_cells": np.nan,
        }

    n_rows, n_columns = df.shape
    total_cells = n_rows * n_columns
    missing_cells = int(df.isna().sum().sum())
    percent_missing = missing_cells / total_cells if total_cells else np.nan

    return {
        "file_name": safe_relative(path),
        "exists": True,
        "n_rows": n_rows,
        "n_columns": n_columns,
        "missing_cells": missing_cells,
        "percent_missing_cells": percent_missing,
    }


def run_pipeline_loss_audit() -> Dict[str, pd.DataFrame]:
    print("\n3. Linkage and pipeline loss audit")
    print("=" * 80)

    files_to_audit = [
        DATA_DIR / "speeches_clean.csv",
        DATA_DIR / "speeches_features.csv",
        DATA_DIR / "speeches_with_sentiment.csv",
        DATA_DIR / "speeches_modeling_30000.csv",
        REPO_ROOT / "feature-engineering" / "engineered_topics.csv",
        REPO_ROOT / "feature-engineering" / "engineered_topics2.csv",
        REPO_ROOT / "besluiten.csv",
        REPO_ROOT / "agendapunten.csv",
        REPO_ROOT / "activiteiten.csv",
    ]

    audit_rows = [audit_file(path) for path in files_to_audit]
    audit_df = pd.DataFrame(audit_rows)
    save_csv(audit_df, "pipeline_file_audit.csv")

    print("\nPipeline file audit:")
    print(audit_df.to_string(index=False))

    topic_comparison_rows = []
    topic1 = REPO_ROOT / "feature-engineering" / "engineered_topics.csv"
    topic2 = REPO_ROOT / "feature-engineering" / "engineered_topics2.csv"

    topic1_info = audit_df.loc[audit_df["file_name"] == safe_relative(topic1)]
    topic2_info = audit_df.loc[audit_df["file_name"] == safe_relative(topic2)]

    if not topic1_info.empty and not topic2_info.empty:
        topic1_exists = bool(topic1_info.iloc[0]["exists"])
        topic2_exists = bool(topic2_info.iloc[0]["exists"])
        if topic1_exists and topic2_exists:
            rows1 = int(topic1_info.iloc[0]["n_rows"])
            rows2 = int(topic2_info.iloc[0]["n_rows"])
            diff = rows1 - rows2
            diff_rate = diff / rows1 if rows1 else np.nan

            topic_comparison_rows.append(
                {
                    "file_a": safe_relative(topic1),
                    "rows_a": rows1,
                    "file_b": safe_relative(topic2),
                    "rows_b": rows2,
                    "row_difference_a_minus_b": diff,
                    "row_difference_rate_of_a": diff_rate,
                    "interpretation": (
                        "Potential topic level data loss. This difference should be checked "
                        "because filtering or joining topic files may remove observations in "
                        "ways that are not random."
                    ),
                }
            )

            print(
                "\nTopic file comparison: "
                f"{safe_relative(topic1)} has {rows1:,} rows and "
                f"{safe_relative(topic2)} has {rows2:,} rows. "
                f"The difference is {diff:,} rows, or {format_rate(diff_rate)} of "
                "the first topic file. This is potential topic level data loss "
                "that should be checked."
            )
        else:
            warn("Could not compare topic files because one or both files do not exist.")

    topic_df = pd.DataFrame(topic_comparison_rows)
    save_csv(topic_df, "topic_file_comparison.csv")

    return {"pipeline_file_audit": audit_df, "topic_file_comparison": topic_df}


# ---------------------------------------------------------------------------
# Section 4. Subgroup outcome audit
# ---------------------------------------------------------------------------

def subgroup_outcome_table(
    df: pd.DataFrame,
    group_col: str,
    min_n: int = MIN_OUTCOME_GROUP_N,
) -> pd.DataFrame:
    output_columns = [
        "group_column",
        "group",
        "n",
        "accepted_count",
        "rejected_count",
        "accepted_rate",
        "rejected_rate",
    ]

    if group_col not in df.columns:
        warn(f"Subgroup outcome audit: group column `{group_col}` is unavailable.")
        return pd.DataFrame(columns=output_columns)

    work = df[[group_col, TARGET_COL]].copy()
    work = work[work[TARGET_COL].notna()]
    work[TARGET_COL] = pd.to_numeric(work[TARGET_COL], errors="coerce")
    work = work[work[TARGET_COL].isin([0, 1])]
    work[group_col] = work[group_col].astype("object").where(work[group_col].notna(), "MISSING_GROUP_VALUE")

    rows = []
    for group_value, group_df in work.groupby(group_col, dropna=False):
        n = len(group_df)
        if n < min_n:
            continue

        accepted_count = int((group_df[TARGET_COL] == 1).sum())
        rejected_count = int((group_df[TARGET_COL] == 0).sum())

        rows.append(
            {
                "group_column": group_col,
                "group": str(group_value),
                "n": n,
                "accepted_count": accepted_count,
                "rejected_count": rejected_count,
                "accepted_rate": accepted_count / n if n else np.nan,
                "rejected_rate": rejected_count / n if n else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values(["accepted_rate", "n"], ascending=[False, False])


def run_subgroup_outcome_audit(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    print("\n4. Subgroup outcome audit")
    print("=" * 80)

    results: Dict[str, pd.DataFrame] = {}

    for group_col in SUBGROUP_OUTCOME_COLUMNS:
        table = subgroup_outcome_table(df, group_col)
        save_csv(table, f"subgroup_outcome_by_{group_col}.csv")
        results[group_col] = table

        print(f"\nSubgroup outcome audit for `{group_col}`:")
        if table.empty:
            print("  No eligible groups after filters.")
        else:
            print(table.head(10).to_string(index=False))

    return results


# ---------------------------------------------------------------------------
# Sections 5 and 6. Model setup and feature set comparison
# ---------------------------------------------------------------------------

def clean_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if TARGET_COL not in work.columns:
        raise ValueError(f"Cannot model without `{TARGET_COL}`.")

    work[TARGET_COL] = pd.to_numeric(work[TARGET_COL], errors="coerce")
    before = len(work)
    work = work[work[TARGET_COL].isin([0, 1])].copy()
    after = len(work)
    if before != after:
        warn(f"Dropped {before - after:,} rows with missing or invalid target values.")

    work[TARGET_COL] = work[TARGET_COL].astype(int)
    return work


def get_train_test_indices(df: pd.DataFrame) -> Tuple[pd.Index, pd.Index]:
    """
    Return train and test indices.
    If the split column is malformed, fall back to a stratified 80 20 split.
    """
    if "split" in df.columns:
        split_values = df["split"].astype(str).str.lower().str.strip()
        train_idx = df.index[split_values == "train"]
        test_idx = df.index[split_values == "test"]
        split_coverage = len(train_idx) + len(test_idx)
        train_share = len(train_idx) / split_coverage if split_coverage else 0.0
        test_share = len(test_idx) / split_coverage if split_coverage else 0.0

        print(
            f"Found existing split column: {len(train_idx):,} train rows and "
            f"{len(test_idx):,} test rows."
        )

        split_is_usable = (
            len(train_idx) >= 100
            and len(test_idx) >= 100
            and train_share >= 0.10
            and test_share >= 0.10
        )

        if split_is_usable:
            print("Using existing split column after validation.")
            return train_idx, test_idx

        warn(
            "Split column exists, but it is not suitable for modelling "
            f"(train share {train_share:.3f}, test share {test_share:.3f}). "
            "Falling back to a stratified 80/20 split."
        )

    y = df[TARGET_COL]
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"Created stratified split: {len(train_idx):,} train rows and {len(test_idx):,} test rows.")
    return pd.Index(train_idx), pd.Index(test_idx)


def build_logistic_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def coerce_features_to_numeric(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    X = df[list(features)].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def train_and_evaluate_model(
    df: pd.DataFrame,
    requested_features: Sequence[str],
    model_name: str,
    train_idx: Optional[pd.Index] = None,
    test_idx: Optional[pd.Index] = None,
) -> Tuple[Dict[str, object], Optional[Pipeline], Optional[pd.DataFrame]]:
    work = clean_model_frame(df)
    features = available_columns(work, requested_features, context=f"{model_name} feature set")

    result_base: Dict[str, object] = {
        "model_name": model_name,
        "requested_features": len(requested_features),
        "used_features": len(features),
        "status": "ok",
        "warning": "",
    }

    empty_metric_keys = [
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "roc_auc",
    ]

    if not features:
        result_base.update({key: np.nan for key in empty_metric_keys})
        result_base["status"] = "skipped"
        result_base["warning"] = "No available features."
        return result_base, None, None

    if train_idx is None or test_idx is None:
        train_idx, test_idx = get_train_test_indices(work)
    else:
        train_idx = train_idx.intersection(work.index)
        test_idx = test_idx.intersection(work.index)
        if len(train_idx) == 0 or len(test_idx) == 0:
            warn(f"{model_name}: supplied split is empty after filtering. Creating a fresh split.")
            train_idx, test_idx = get_train_test_indices(work)

    X_train = coerce_features_to_numeric(work.loc[train_idx], features)
    X_test = coerce_features_to_numeric(work.loc[test_idx], features)
    y_train = work.loc[train_idx, TARGET_COL]
    y_test = work.loc[test_idx, TARGET_COL]

    if y_train.nunique() < 2:
        result_base.update({key: np.nan for key in empty_metric_keys})
        result_base["status"] = "skipped"
        result_base["warning"] = "Training data has fewer than two outcome classes."
        return result_base, None, None

    pipeline = build_logistic_pipeline()

    try:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_score = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
        metrics = metric_dict(y_test, y_pred, y_score)
        result_base.update(metrics)
        result_base["train_n"] = len(train_idx)
        result_base["test_n"] = len(test_idx)

        prediction_frame = work.loc[test_idx].copy()
        prediction_frame["_y_true"] = y_test.values
        prediction_frame["_y_pred"] = y_pred
        prediction_frame["_y_score"] = y_score if y_score is not None else np.nan
        prediction_frame["_model_name"] = model_name

        return result_base, pipeline, prediction_frame

    except Exception as exc:
        warn(f"{model_name}: model training or evaluation failed: {exc}")
        result_base.update({key: np.nan for key in empty_metric_keys})
        result_base["status"] = "failed"
        result_base["warning"] = str(exc)
        return result_base, None, None


def run_feature_set_comparison(df: pd.DataFrame) -> Dict[str, object]:
    print("\n5. Model setup")
    print("=" * 80)

    model_df = clean_model_frame(df)
    train_idx, test_idx = get_train_test_indices(model_df)

    model_specs = {
        "Model A structural temporal only": MODEL_A_FEATURES,
        "Model B add speech and sentiment": MODEL_B_FEATURES,
        "Model C add party indicators": MODEL_C_FEATURES,
    }

    print("\n6. Feature set comparison")
    print("=" * 80)

    rows = []
    fitted_models: Dict[str, Pipeline] = {}
    prediction_frames: Dict[str, pd.DataFrame] = {}

    for model_name, features in model_specs.items():
        print(f"\nTraining {model_name}")
        row, model, pred_frame = train_and_evaluate_model(
            model_df,
            features,
            model_name,
            train_idx=train_idx,
            test_idx=test_idx,
        )
        rows.append(row)

        if model is not None:
            fitted_models[model_name] = model
        if pred_frame is not None:
            prediction_frames[model_name] = pred_frame

    comparison = pd.DataFrame(rows)
    save_csv(comparison, "feature_set_model_comparison.csv")

    print("\nFeature set comparison:")
    print(comparison.to_string(index=False))

    model_a = comparison.loc[comparison["model_name"] == "Model A structural temporal only"]
    model_b = comparison.loc[comparison["model_name"] == "Model B add speech and sentiment"]
    model_c = comparison.loc[comparison["model_name"] == "Model C add party indicators"]

    print(
        "\nInterpretation: If Model C improves strongly over Model A or B, "
        "that supports the omitted variable bias hypothesis because political "
        "party indicators add explanatory power beyond visible debate indicators."
    )

    if not model_a.empty and not model_c.empty:
        a_f1 = model_a.iloc[0].get("f1_macro", np.nan)
        c_f1 = model_c.iloc[0].get("f1_macro", np.nan)
        if pd.notna(a_f1) and pd.notna(c_f1):
            print(f"Observed macro F1 change from Model A to Model C: {c_f1 - a_f1:.4f}")

    if not model_b.empty and not model_c.empty:
        b_f1 = model_b.iloc[0].get("f1_macro", np.nan)
        c_f1 = model_c.iloc[0].get("f1_macro", np.nan)
        if pd.notna(b_f1) and pd.notna(c_f1):
            print(f"Observed macro F1 change from Model B to Model C: {c_f1 - b_f1:.4f}")

    return {
        "comparison": comparison,
        "fitted_models": fitted_models,
        "prediction_frames": prediction_frames,
        "train_idx": train_idx,
        "test_idx": test_idx,
    }


# ---------------------------------------------------------------------------
# Section 7. Subgroup performance audit
# ---------------------------------------------------------------------------

def subgroup_performance_table(
    prediction_frame: pd.DataFrame,
    group_cols: Sequence[str],
    min_n: int = MIN_PERFORMANCE_GROUP_N,
) -> pd.DataFrame:
    rows = []

    for group_col in group_cols:
        if group_col not in prediction_frame.columns:
            warn(f"Subgroup performance audit: group column `{group_col}` is unavailable.")
            continue

        work = prediction_frame[[group_col, "_y_true", "_y_pred"]].copy()
        work[group_col] = work[group_col].astype("object").where(work[group_col].notna(), "MISSING_GROUP_VALUE")

        for group_value, group_df in work.groupby(group_col, dropna=False):
            n = len(group_df)
            if n < min_n:
                continue
            if group_df["_y_true"].nunique() < 2:
                continue

            metrics = metric_dict(group_df["_y_true"], group_df["_y_pred"], y_score=None)
            rows.append(
                {
                    "group_column": group_col,
                    "group_value": str(group_value),
                    "n": n,
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "precision_macro": metrics["precision_macro"],
                    "recall_macro": metrics["recall_macro"],
                    "f1_macro": metrics["f1_macro"],
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "group_column",
                "group_value",
                "n",
                "accuracy",
                "balanced_accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
            ]
        )

    return pd.DataFrame(rows).sort_values(["group_column", "f1_macro"], ascending=[True, True])


def choose_prediction_frame(model_results: Dict[str, object]) -> Tuple[str, Optional[pd.DataFrame]]:
    prediction_frames: Dict[str, pd.DataFrame] = model_results["prediction_frames"]

    if "Model C add party indicators" in prediction_frames:
        return "Model C add party indicators", prediction_frames["Model C add party indicators"]

    comparison: pd.DataFrame = model_results["comparison"]
    ok = comparison[comparison["status"] == "ok"].copy()
    if ok.empty:
        return "none", None

    ok = ok.sort_values("f1_macro", ascending=False)
    best_name = str(ok.iloc[0]["model_name"])
    return best_name, prediction_frames.get(best_name)


def run_subgroup_performance_audit(model_results: Dict[str, object]) -> pd.DataFrame:
    print("\n7. Subgroup performance audit")
    print("=" * 80)

    model_name, prediction_frame = choose_prediction_frame(model_results)

    if prediction_frame is None:
        warn("No prediction frame is available for subgroup performance audit.")
        table = pd.DataFrame(
            columns=[
                "group_column",
                "group_value",
                "n",
                "accuracy",
                "balanced_accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
            ]
        )
        save_csv(table, "subgroup_performance_audit.csv")
        return table

    print(f"Using {model_name} for subgroup performance audit.")

    table = subgroup_performance_table(prediction_frame, SUBGROUP_PERFORMANCE_COLUMNS)
    save_csv(table, "subgroup_performance_audit.csv")

    if table.empty:
        print("No eligible subgroup performance rows after filters.")
    else:
        print(table.head(20).to_string(index=False))

    return table


# ---------------------------------------------------------------------------
# Section 8. Robustness checks
# ---------------------------------------------------------------------------

def run_robustness_checks(df: pd.DataFrame) -> pd.DataFrame:
    print("\n8. Robustness checks")
    print("=" * 80)

    model_df = clean_model_frame(df)

    checks = []

    specs = [
        (
            "Model B without sentiment columns",
            model_df,
            [feature for feature in MODEL_B_FEATURES if feature not in SENTIMENT_FEATURES],
        ),
        (
            "Model B with sentiment columns",
            model_df,
            MODEL_B_FEATURES,
        ),
        (
            "Model C with party indicators",
            model_df,
            MODEL_C_FEATURES,
        ),
    ]

    fully_missing_cols = [col for col in model_df.columns if model_df[col].isna().all()]
    model_df_no_full_missing = model_df.drop(columns=fully_missing_cols, errors="ignore")
    specs.append(
        (
            "Model C after removing fully missing columns from the dataset",
            model_df_no_full_missing,
            MODEL_C_FEATURES,
        )
    )

    if "speech_duration_seconds" in model_df.columns:
        duration = pd.to_numeric(model_df["speech_duration_seconds"], errors="coerce")
        threshold = duration.quantile(0.99)
        if pd.notna(threshold):
            filtered = model_df[(duration.isna()) | (duration <= threshold)].copy()
            specs.append(
                (
                    "Model C after filtering speech_duration_seconds above the 99th percentile",
                    filtered,
                    MODEL_C_FEATURES,
                )
            )
            print(
                "Speech duration robustness filter: "
                f"99th percentile threshold is {threshold:.3f}; "
                f"kept {len(filtered):,} of {len(model_df):,} rows."
            )
        else:
            warn("speech_duration_seconds exists but has no numeric values. Skipping duration filter check.")
    else:
        warn("speech_duration_seconds is unavailable. Skipping duration filter check.")

    for check_name, check_df, features in specs:
        print(f"\nRunning robustness check: {check_name}")
        row, _, _ = train_and_evaluate_model(check_df, features, check_name)
        row["fully_missing_columns_removed_count"] = len(fully_missing_cols) if "fully missing" in check_name else 0
        checks.append(row)

    robustness = pd.DataFrame(checks)
    save_csv(robustness, "robustness_checks.csv")

    print("\nRobustness checks:")
    print(robustness.to_string(index=False))

    return robustness


# ---------------------------------------------------------------------------
# Section 9. Outcome class performance audit
# ---------------------------------------------------------------------------

def run_outcome_class_performance_audit(model_results: Dict[str, object]) -> Dict[str, pd.DataFrame]:
    print("\n9. Outcome class performance audit")
    print("=" * 80)

    prediction_frames: Dict[str, pd.DataFrame] = model_results.get("prediction_frames", {})

    class_rows = []
    matrix_rows = []

    for model_name, pred in prediction_frames.items():
        if pred is None or pred.empty:
            continue

        y_true = pred["_y_true"]
        y_pred = pred["_y_pred"]

        report = classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        )

        label_names = {
            "0": "rejected",
            "1": "accepted",
        }

        for label_value, label_name in label_names.items():
            if label_value in report:
                row = report[label_value]
                class_rows.append(
                    {
                        "model_name": model_name,
                        "class_value": label_value,
                        "class_label": label_name,
                        "precision": row.get("precision", np.nan),
                        "recall": row.get("recall", np.nan),
                        "f1_score": row.get("f1-score", np.nan),
                        "support": row.get("support", np.nan),
                    }
                )

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        matrix_rows.append(
            {
                "model_name": model_name,
                "true_rejected_pred_rejected": int(cm[0, 0]),
                "true_rejected_pred_accepted": int(cm[0, 1]),
                "true_accepted_pred_rejected": int(cm[1, 0]),
                "true_accepted_pred_accepted": int(cm[1, 1]),
            }
        )

    class_table = pd.DataFrame(class_rows)
    matrix_table = pd.DataFrame(matrix_rows)

    save_csv(class_table, "outcome_class_performance.csv")
    save_csv(matrix_table, "confusion_matrix_by_model.csv")

    if class_table.empty:
        print("No class level performance table could be created.")
    else:
        print("\nOutcome class performance:")
        print(class_table.to_string(index=False))

    if matrix_table.empty:
        print("No confusion matrix table could be created.")
    else:
        print("\nConfusion matrix by model:")
        print(matrix_table.to_string(index=False))

    return {
        "outcome_class_performance": class_table,
        "confusion_matrix_by_model": matrix_table,
    }


# ---------------------------------------------------------------------------
# Section 10. Bias hypothesis and method summary tables
# ---------------------------------------------------------------------------

def max_acceptance_spread(subgroup_outcomes: Dict[str, pd.DataFrame], group_col: str) -> str:
    table = subgroup_outcomes.get(group_col, pd.DataFrame())
    if table is None or table.empty or "accepted_rate" not in table.columns:
        return "No eligible subgroup outcome table."

    high = table.sort_values("accepted_rate", ascending=False).iloc[0]
    low = table.sort_values("accepted_rate", ascending=True).iloc[0]
    spread = float(high["accepted_rate"] - low["accepted_rate"])
    return (
        f"Accepted rate ranged from {format_rate(low['accepted_rate'])} for "
        f"{low['group']} to {format_rate(high['accepted_rate'])} for "
        f"{high['group']}. Spread was {format_rate(spread)}."
    )


def get_model_delta(model_results: Dict[str, object]) -> str:
    comparison = model_results.get("comparison", pd.DataFrame())
    if comparison is None or comparison.empty:
        return "No model comparison available."

    ok = comparison[comparison["status"] == "ok"].copy()
    if ok.empty:
        return "No completed model comparison available."

    lookup = ok.set_index("model_name")

    needed = [
        "Model A structural temporal only",
        "Model C add party indicators",
    ]

    if all(name in lookup.index for name in needed):
        a = lookup.loc["Model A structural temporal only", "f1_macro"]
        c = lookup.loc["Model C add party indicators", "f1_macro"]
        return (
            f"Macro F1 increased from {a:.4f} in Model A to {c:.4f} in Model C. "
            f"Difference was {c - a:.4f}."
        )

    return "Model A and Model C were not both available."


def get_class_performance_evidence(class_results: Dict[str, pd.DataFrame]) -> str:
    class_table = class_results.get("outcome_class_performance", pd.DataFrame())
    if class_table is None or class_table.empty:
        return "No outcome class performance table available."

    model_c = class_table[class_table["model_name"] == "Model C add party indicators"].copy()
    if model_c.empty:
        model_c = class_table.copy()

    pieces = []
    for _, row in model_c.iterrows():
        pieces.append(
            f"{row['class_label']} recall {row['recall']:.4f}, "
            f"F1 {row['f1_score']:.4f}"
        )

    return "; ".join(pieces)


def write_bias_hypothesis_summary(
    df: pd.DataFrame,
    pipeline_results: Dict[str, pd.DataFrame],
    subgroup_outcomes: Dict[str, pd.DataFrame],
    model_results: Dict[str, object],
    class_results: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    print("\n10. Bias hypothesis summary table")
    print("=" * 80)

    speaker_party_missing = df["speaker_party"].isna().mean() if "speaker_party" in df.columns else np.nan
    aanvang_missing = df["Aanvangstijd_y"].isna().mean() if "Aanvangstijd_y" in df.columns else np.nan
    eind_missing = df["Eindtijd_y"].isna().mean() if "Eindtijd_y" in df.columns else np.nan

    topic = pipeline_results.get("topic_file_comparison", pd.DataFrame())
    if topic is not None and not topic.empty:
        topic_loss = topic.iloc[0]["row_difference_rate_of_a"]
        topic_evidence = (
            f"Topic feature file lost {format_rate(topic_loss)} of rows "
            "between engineered_topics.csv and engineered_topics2.csv."
        )
    else:
        topic_evidence = "No topic file comparison available."

    rows = [
        {
            "bias_hypothesis": "Data linkage and missingness bias",
            "basis_for_hypothesis": (
                "The project depends on linked Tweede Kamer tables. Failed links or missing fields "
                "can remove cases before modelling."
            ),
            "coded_test_used": (
                "Overall missingness audit, missingness by subgroup, pipeline file audit, "
                "topic file comparison."
            ),
            "evidence_from_this_run": (
                f"Aanvangstijd_y missingness was {format_rate(aanvang_missing)}. "
                f"Eindtijd_y missingness was {format_rate(eind_missing)}. "
                f"speaker_party missingness was {format_rate(speaker_party_missing)}. "
                f"{topic_evidence}"
            ),
            "audit_status": "Risk detected",
            "mitigation_if_present": (
                "Do not silently drop missing cases. Document row loss, use cleaned replacement "
                "variables where justified, add missingness flags, and compare retained versus "
                "removed cases."
            ),
            "main_output_files": (
                "missingness_overall.csv; missingness_by_group files; "
                "pipeline_file_audit.csv; topic_file_comparison.csv"
            ),
        },
        {
            "bias_hypothesis": "Omitted variable bias",
            "basis_for_hypothesis": (
                "Literature and interviews suggest that party position, institutional role, "
                "agenda access, and political context may matter more than visible debate features."
            ),
            "coded_test_used": (
                "Feature set comparison between structural temporal features, speech sentiment "
                "features, and party indicators."
            ),
            "evidence_from_this_run": get_model_delta(model_results),
            "audit_status": "Risk partly supported",
            "mitigation_if_present": (
                "Add political controls where available, such as party indicators, proposer identity, "
                "government or opposition status, motion type, policy area, and committee role."
            ),
            "main_output_files": "feature_set_model_comparison.csv; robustness_checks.csv",
        },
        {
            "bias_hypothesis": "Feature bias",
            "basis_for_hypothesis": (
                "Speech duration, sentiment, speaker frequency, and timing are visible indicators. "
                "They may reflect deeper mechanisms rather than direct causes of outcomes."
            ),
            "coded_test_used": (
                "Subgroup outcome audit by time bin, sentiment label, speaker role, and long speech flag."
            ),
            "evidence_from_this_run": (
                f"Time bin pattern: {max_acceptance_spread(subgroup_outcomes, 'time_bin')} "
                f"Sentiment pattern: {max_acceptance_spread(subgroup_outcomes, 'sentiment_label')}"
            ),
            "audit_status": "Risk detected as patterned association",
            "mitigation_if_present": (
                "Interpret variables as visible indicators of agenda access, political visibility, "
                "public performance, scheduling constraints, and institutional power, not as direct "
                "causal predictors."
            ),
            "main_output_files": (
                "subgroup_outcome_by_time_bin.csv; subgroup_outcome_by_sentiment_label.csv; "
                "subgroup_outcome_by_long_speech_flag.csv"
            ),
        },
        {
            "bias_hypothesis": "Visibility bias",
            "basis_for_hypothesis": (
                "The dataset captures public formal parliamentary debate but not informal negotiation, "
                "party discipline, committee bargaining, amendments, lobbying, or stakeholder influence."
            ),
            "coded_test_used": (
                "Compare whether visible debate variables alone perform weakly, then check whether party "
                "context improves performance."
            ),
            "evidence_from_this_run": get_model_delta(model_results),
            "audit_status": "Risk supported by limited model performance",
            "mitigation_if_present": (
                "State clearly that the model audits the visible parliamentary record only. Do not claim "
                "that it fully explains decision making."
            ),
            "main_output_files": "feature_set_model_comparison.csv; week8_bias_audit_summary.txt",
        },
        {
            "bias_hypothesis": "Outcome imbalance bias",
            "basis_for_hypothesis": (
                "Even when the overall target distribution is balanced, the model may perform differently "
                "for accepted and rejected motions or across subgroups."
            ),
            "coded_test_used": (
                "Outcome class performance audit, confusion matrix, macro F1, balanced accuracy, "
                "subgroup performance audit."
            ),
            "evidence_from_this_run": get_class_performance_evidence(class_results),
            "audit_status": "Risk checked directly",
            "mitigation_if_present": (
                "Report class level performance, macro F1, balanced accuracy, and subgroup metrics rather "
                "than relying only on overall accuracy."
            ),
            "main_output_files": (
                "outcome_class_performance.csv; confusion_matrix_by_model.csv; "
                "subgroup_performance_audit.csv"
            ),
        },
    ]

    table = pd.DataFrame(rows)
    save_csv(table, "bias_hypothesis_summary.csv")

    print(table[["bias_hypothesis", "audit_status", "evidence_from_this_run"]].to_string(index=False))
    return table


def write_audit_method_summary() -> pd.DataFrame:
    print("\n11. Audit method summary table")
    print("=" * 80)

    rows = [
        {
            "week_goal_requirement": "Systematic bias audit strategy",
            "coded_component": "Full script sections run in fixed order",
            "output_file": "week8_bias_audit_summary.txt",
            "covered": "yes",
        },
        {
            "week_goal_requirement": "Identified biases",
            "coded_component": "Bias hypothesis summary table",
            "output_file": "bias_hypothesis_summary.csv",
            "covered": "yes",
        },
        {
            "week_goal_requirement": "Testing methods used",
            "coded_component": (
                "Missingness audit, linkage audit, subgroup outcome audit, feature set comparison, "
                "class performance audit, subgroup performance audit, robustness checks"
            ),
            "output_file": (
                "missingness_overall.csv; pipeline_file_audit.csv; subgroup outcome files; "
                "feature_set_model_comparison.csv; outcome_class_performance.csv; "
                "subgroup_performance_audit.csv; robustness_checks.csv"
            ),
            "covered": "yes",
        },
        {
            "week_goal_requirement": "Bias mitigation strategies",
            "coded_component": "Mitigation column in bias hypothesis summary",
            "output_file": "bias_hypothesis_summary.csv",
            "covered": "yes",
        },
        {
            "week_goal_requirement": "Connection to data cleaning",
            "coded_component": "Missingness audit and pipeline row loss audit",
            "output_file": "missingness_overall.csv; pipeline_file_audit.csv; topic_file_comparison.csv",
            "covered": "yes",
        },
        {
            "week_goal_requirement": "Stand Up Report evidence",
            "coded_component": "Summary text plus presentation ready tables",
            "output_file": "week8_bias_audit_summary.txt; bias_hypothesis_summary.csv",
            "covered": "yes",
        },
    ]

    table = pd.DataFrame(rows)
    save_csv(table, "audit_method_summary.csv")
    print(table.to_string(index=False))
    return table


# ---------------------------------------------------------------------------
# Section 12. Written summary
# ---------------------------------------------------------------------------

def summarize_missingness(overall_missingness: pd.DataFrame, df: pd.DataFrame) -> str:
    lines = []

    top_missing = overall_missingness.head(10)
    lines.append("Main missingness findings")
    lines.append("")

    if top_missing.empty:
        lines.append("No missingness table was available.")
    else:
        lines.append("Columns with the highest missingness rates were:")
        for _, row in top_missing.iterrows():
            lines.append(
                f"* {row['column']}: {int(row['missing_count']):,} missing values, "
                f"{format_rate(float(row['missing_rate']))} missing."
            )

    for col in ["Aanvangstijd_y", "Eindtijd_y", "speaker_party"]:
        if col in df.columns:
            lines.append(
                f"* {col}: {int(df[col].isna().sum()):,} missing values, "
                f"{format_rate(df[col].isna().mean())} missing."
            )
        else:
            lines.append(f"* {col}: not available in the dataset.")

    lines.append("")
    lines.append(
        "Aanvangstijd_y and Eindtijd_y are important because they connect cleaning and linkage "
        "decisions to possible downstream bias. speaker_party is important because party is a central "
        "political context variable."
    )

    return "\n".join(lines)


def summarize_subgroup_outcomes(subgroup_tables: Dict[str, pd.DataFrame]) -> str:
    lines = []
    lines.append("Main subgroup outcome differences")
    lines.append("")

    any_table = False

    for group_col, table in subgroup_tables.items():
        if table is None or table.empty:
            continue

        any_table = True
        high = table.sort_values("accepted_rate", ascending=False).iloc[0]
        low = table.sort_values("accepted_rate", ascending=True).iloc[0]
        lines.append(
            f"* {group_col}: highest accepted rate was {format_rate(high['accepted_rate'])} "
            f"for `{high['group']}` with n={int(high['n']):,}; lowest accepted rate was "
            f"{format_rate(low['accepted_rate'])} for `{low['group']}` with n={int(low['n']):,}."
        )

    if not any_table:
        lines.append("No subgroup outcome tables had eligible groups after the minimum size filter.")

    return "\n".join(lines)


def summarize_feature_sets(comparison: pd.DataFrame) -> str:
    lines = []
    lines.append("Feature set model comparison")
    lines.append("")

    ok = comparison[comparison["status"] == "ok"].copy() if "status" in comparison.columns else pd.DataFrame()

    if ok.empty:
        lines.append("No feature set model comparison was available.")
        return "\n".join(lines)

    for _, row in ok.iterrows():
        lines.append(
            f"* {row['model_name']}: balanced accuracy={row['balanced_accuracy']:.4f}, "
            f"macro F1={row['f1_macro']:.4f}, ROC AUC={row['roc_auc']:.4f}."
        )

    lookup = ok.set_index("model_name")
    if "Model A structural temporal only" in lookup.index and "Model C add party indicators" in lookup.index:
        delta = lookup.loc["Model C add party indicators", "f1_macro"] - lookup.loc[
            "Model A structural temporal only",
            "f1_macro",
        ]
        lines.append(
            f"* Macro F1 difference from Model A to Model C was {delta:.4f}. "
            "A strong improvement after adding party indicators would support the omitted variable bias hypothesis."
        )

    if "Model B add speech and sentiment" in lookup.index and "Model C add party indicators" in lookup.index:
        delta = lookup.loc["Model C add party indicators", "f1_macro"] - lookup.loc[
            "Model B add speech and sentiment",
            "f1_macro",
        ]
        lines.append(f"* Macro F1 difference from Model B to Model C was {delta:.4f}.")

    return "\n".join(lines)


def summarize_subgroup_performance(performance: pd.DataFrame) -> str:
    lines = []
    lines.append("Subgroup performance and uneven model behaviour")
    lines.append("")

    if performance is None or performance.empty:
        lines.append("No eligible subgroup performance rows were available after filters.")
        return "\n".join(lines)

    for group_col, group_df in performance.groupby("group_column"):
        best = group_df.sort_values("f1_macro", ascending=False).iloc[0]
        worst = group_df.sort_values("f1_macro", ascending=True).iloc[0]
        spread = best["f1_macro"] - worst["f1_macro"]

        lines.append(
            f"* {group_col}: macro F1 ranged from {worst['f1_macro']:.4f} "
            f"for `{worst['group_value']}` with n={int(worst['n']):,} to "
            f"{best['f1_macro']:.4f} for `{best['group_value']}` with n={int(best['n']):,}; "
            f"spread={spread:.4f}."
        )

    lines.append(
        "* Large subgroup performance spreads should be interpreted as signs of possible uneven model behaviour, "
        "especially when they align with politically meaningful groups or known missingness patterns."
    )

    return "\n".join(lines)


def summarize_pipeline_loss(pipeline_results: Dict[str, pd.DataFrame]) -> str:
    lines = []
    lines.append("Data linkage and pipeline loss")
    lines.append("")

    audit = pipeline_results.get("pipeline_file_audit", pd.DataFrame())
    if audit.empty:
        lines.append("No pipeline file audit was available.")
    else:
        existing = audit[audit["exists"] == True].copy()
        missing = audit[audit["exists"] == False].copy()

        if not existing.empty:
            lines.append("Existing audited files and row counts:")
            for _, row in existing.iterrows():
                lines.append(
                    f"* {row['file_name']}: {format_int(row['n_rows'])} rows, "
                    f"{format_int(row['n_columns'])} columns, "
                    f"{format_rate(row['percent_missing_cells'])} missing cells."
                )

        if not missing.empty:
            lines.append("Unavailable files:")
            for _, row in missing.iterrows():
                lines.append(f"* {row['file_name']}")

    topic = pipeline_results.get("topic_file_comparison", pd.DataFrame())
    if topic is not None and not topic.empty:
        row = topic.iloc[0]
        lines.append(
            f"* Topic file comparison found {format_int(row['row_difference_a_minus_b'])} fewer rows "
            "in the second topic file than in the first. This should be checked as potential topic level data loss."
        )

    return "\n".join(lines)


def summarize_robustness(robustness: pd.DataFrame) -> str:
    lines = []
    lines.append("Robustness checks")
    lines.append("")

    if robustness is None or robustness.empty:
        lines.append("No robustness checks were available.")
        return "\n".join(lines)

    ok = robustness[robustness["status"] == "ok"].copy() if "status" in robustness.columns else pd.DataFrame()
    if ok.empty:
        lines.append("No robustness checks completed successfully.")
    else:
        for _, row in ok.iterrows():
            lines.append(
                f"* {row['model_name']}: balanced accuracy={row['balanced_accuracy']:.4f}, "
                f"macro F1={row['f1_macro']:.4f}, ROC AUC={row['roc_auc']:.4f}."
            )

    skipped = robustness[robustness["status"] != "ok"].copy() if "status" in robustness.columns else pd.DataFrame()
    if not skipped.empty:
        lines.append("Skipped or failed robustness checks:")
        for _, row in skipped.iterrows():
            lines.append(f"* {row['model_name']}: {row.get('warning', '')}")

    return "\n".join(lines)


def summarize_class_performance(class_results: Dict[str, pd.DataFrame]) -> str:
    lines = []
    lines.append("Outcome class performance")
    lines.append("")

    class_table = class_results.get("outcome_class_performance", pd.DataFrame())
    if class_table is None or class_table.empty:
        lines.append("No outcome class performance table was available.")
        return "\n".join(lines)

    for model_name, model_df in class_table.groupby("model_name"):
        pieces = []
        for _, row in model_df.iterrows():
            pieces.append(
                f"{row['class_label']}: precision={row['precision']:.4f}, "
                f"recall={row['recall']:.4f}, F1={row['f1_score']:.4f}"
            )
        lines.append(f"* {model_name}: " + "; ".join(pieces))

    return "\n".join(lines)


def write_summary(
    df: pd.DataFrame,
    missingness_results: Dict[str, pd.DataFrame],
    pipeline_results: Dict[str, pd.DataFrame],
    subgroup_outcomes: Dict[str, pd.DataFrame],
    model_results: Dict[str, object],
    subgroup_performance: pd.DataFrame,
    robustness: pd.DataFrame,
    class_results: Dict[str, pd.DataFrame],
) -> Path:
    print("\n12. Write text summary")
    print("=" * 80)

    overall_missingness = missingness_results.get("overall", pd.DataFrame())
    comparison = model_results.get("comparison", pd.DataFrame())

    sections = [
        "Week 8 Systematic Bias Audit Summary",
        "=" * 80,
        "",
        summarize_missingness(overall_missingness, df),
        "",
        summarize_pipeline_loss(pipeline_results),
        "",
        summarize_subgroup_outcomes(subgroup_outcomes),
        "",
        summarize_feature_sets(comparison),
        "",
        summarize_class_performance(class_results),
        "",
        summarize_subgroup_performance(subgroup_performance),
        "",
        summarize_robustness(robustness),
        "",
        "Connection to bias hypotheses",
        "",
        "These results do not prove bias by themselves. They are audit evidence showing where bias risks may exist.",
        "",
        "The five bias hypotheses are:",
        "* omitted variable bias",
        "* feature bias",
        "* visibility bias",
        "* outcome imbalance bias",
        "* data linkage and missingness bias",
        "",
        (
            "The current variables should not be interpreted as direct causes of parliamentary outcomes. "
            "They are visible indicators of deeper political mechanisms, such as agenda access, political "
            "visibility, public performance, scheduling constraints, and institutional power."
        ),
        "",
        "Interpretive guide:",
        "* Omitted variable bias is suggested when adding political party indicators changes performance strongly.",
        "* Feature bias is suggested when engineered features contain systematic missingness or measurement choices.",
        "* Visibility bias is suggested when speaker role, speech length, or frequency features dominate patterns.",
        "* Outcome imbalance bias is suggested when accepted and rejected outcomes are uneven in subgroups.",
        "* Data linkage and missingness bias is suggested when files lose rows or key fields are missing in patterned ways.",
        "",
    ]

    summary_text = "\n".join(sections)
    path = OUTPUT_DIR / "week8_bias_audit_summary.txt"
    path.write_text(summary_text, encoding="utf-8")
    print(f"Saved {safe_relative(path)}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Week 8 Systematic Bias Audit")
    print("=" * 80)
    print("This script runs locally on CPU and does not rerun sentiment analysis.")

    ensure_output_dir()

    df = load_main_dataset()
    missingness_results = run_missingness_audit(df)
    pipeline_results = run_pipeline_loss_audit()
    subgroup_outcomes = run_subgroup_outcome_audit(df)
    model_results = run_feature_set_comparison(df)
    subgroup_performance = run_subgroup_performance_audit(model_results)
    robustness = run_robustness_checks(df)
    class_results = run_outcome_class_performance_audit(model_results)

    write_bias_hypothesis_summary(
        df=df,
        pipeline_results=pipeline_results,
        subgroup_outcomes=subgroup_outcomes,
        model_results=model_results,
        class_results=class_results,
    )

    write_audit_method_summary()

    write_summary(
        df=df,
        missingness_results=missingness_results,
        pipeline_results=pipeline_results,
        subgroup_outcomes=subgroup_outcomes,
        model_results=model_results,
        subgroup_performance=subgroup_performance,
        robustness=robustness,
        class_results=class_results,
    )

    print("\n13. Completion")
    print("=" * 80)
    print("Audit complete.")
    print(f"All available outputs were saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
