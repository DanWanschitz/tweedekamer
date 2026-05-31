"""
Week 8 Bias Mitigation Implementation

This script takes the bias audit results one step further by implementing
practical mitigation steps.

It does not claim to remove all bias. Some risks, especially visibility bias,
cannot be solved fully with the available public parliamentary data.

Outputs are written to:

    02_bias_mitigation_outputs
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import json
import warnings

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
TARGET_COL = "label"
MIN_GROUP_N = 100


def find_repo_root() -> Path:
    expected = Path("parliamentary_notebooks_speeches") / "speeches_with_sentiment.csv"

    candidates = []
    cwd = Path.cwd().resolve()
    candidates.append(cwd)
    candidates.extend(cwd.parents)

    try:
        script_dir = Path(__file__).resolve().parent
        candidates.append(script_dir)
        candidates.extend(script_dir.parents)
    except NameError:
        pass

    home = Path.home()
    candidates.extend(
        [
            home / "Desktop" / "tweedekamer",
            home / "desktop" / "tweedekamer",
            home / "Documents" / "tweedekamer",
            home,
        ]
    )

    seen = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate not in seen:
            seen.append(candidate)

    for candidate in seen:
        if (candidate / expected).exists():
            return candidate

    return cwd


REPO_ROOT = find_repo_root()
DATA_PATH = REPO_ROOT / "parliamentary_notebooks_speeches" / "speeches_with_sentiment.csv"
OUTPUT_DIR = REPO_ROOT / "02_bias_mitigation_outputs"


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

MODEL_B_FEATURES = MODEL_A_FEATURES + [
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

MITIGATION_FEATURES = [
    "speaker_party_missing",
    "raw_agenda_time_missing",
    "sentiment_available",
    "speech_duration_extreme",
    "topic_time_proxy_available",
]


MODEL_C_FEATURES = MODEL_B_FEATURES + PARTY_FEATURES
MODEL_D_MITIGATED_FEATURES = MODEL_C_FEATURES + MITIGATION_FEATURES


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def warn(message: str) -> None:
    print(f"WARNING: {message}")
    warnings.warn(message, stacklevel=2)


def save_csv(df: pd.DataFrame, filename: str) -> Path:
    ensure_output_dir()
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    print(f"Saved {path}")
    return path


def available_columns(df: pd.DataFrame, columns: Sequence[str], context: str) -> List[str]:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        warn(f"{context}: skipping unavailable columns {missing}")
    return [col for col in columns if col in df.columns]


def load_data() -> pd.DataFrame:
    print("Loading data")
    print("=" * 80)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find dataset at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, low_memory=False)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column {TARGET_COL} is missing.")

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df[df[TARGET_COL].isin([0, 1])].copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    print(f"Loaded {DATA_PATH}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.shape[1]:,}")
    print(df[TARGET_COL].value_counts().sort_index())

    return df


def create_mitigated_dataset(df: pd.DataFrame) -> pd.DataFrame:
    print("\nCreating mitigation features")
    print("=" * 80)

    work = df.copy()

    if "speaker_party" in work.columns:
        work["speaker_party_missing"] = work["speaker_party"].isna().astype(int)
    else:
        work["speaker_party_missing"] = 1

    raw_time_cols = [col for col in ["Aanvangstijd_y", "Eindtijd_y"] if col in work.columns]
    if raw_time_cols:
        work["raw_agenda_time_missing"] = work[raw_time_cols].isna().all(axis=1).astype(int)
    else:
        work["raw_agenda_time_missing"] = 1

    sentiment_cols = [col for col in ["sentiment_score", "sentiment_pos", "sentiment_neg"] if col in work.columns]
    if sentiment_cols:
        work["sentiment_available"] = work[sentiment_cols].notna().all(axis=1).astype(int)
    else:
        work["sentiment_available"] = 0

    if "speech_duration_seconds" in work.columns:
        duration = pd.to_numeric(work["speech_duration_seconds"], errors="coerce")
        threshold = duration.quantile(0.99)
        work["speech_duration_extreme"] = ((duration > threshold) & duration.notna()).astype(int)
    else:
        work["speech_duration_extreme"] = 0

    topic_proxy_cols = [col for col in ["time_diff", "hour", "time_bin_ordinal"] if col in work.columns]
    if topic_proxy_cols:
        work["topic_time_proxy_available"] = work[topic_proxy_cols].notna().any(axis=1).astype(int)
    else:
        work["topic_time_proxy_available"] = 0

    fully_missing_cols = [col for col in work.columns if work[col].isna().all()]
    work = work.drop(columns=fully_missing_cols, errors="ignore")

    mitigation_log = pd.DataFrame(
        [
            {
                "mitigation_step": "Add speaker party missingness flag",
                "reason": "speaker_party had notable missingness in the audit",
                "implemented_column": "speaker_party_missing",
            },
            {
                "mitigation_step": "Add raw agenda time missingness flag",
                "reason": "Aanvangstijd_y and Eindtijd_y were fully missing in the audit",
                "implemented_column": "raw_agenda_time_missing",
            },
            {
                "mitigation_step": "Add sentiment availability flag",
                "reason": "sentiment is an engineered feature and should be checked as available",
                "implemented_column": "sentiment_available",
            },
            {
                "mitigation_step": "Add extreme speech duration flag",
                "reason": "long speeches may indicate agenda access and visibility",
                "implemented_column": "speech_duration_extreme",
            },
            {
                "mitigation_step": "Remove fully missing columns",
                "reason": "fully missing columns cannot contribute direct predictive information",
                "implemented_column": ", ".join(fully_missing_cols) if fully_missing_cols else "none",
            },
        ]
    )

    save_csv(mitigation_log, "mitigation_steps_log.csv")
    save_csv(work.head(200), "mitigated_dataset_preview.csv")

    print(f"Removed fully missing columns: {fully_missing_cols}")
    print("Created mitigation indicator columns.")

    return work


def build_pipeline() -> Pipeline:
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


def coerce_features(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    X = df[list(features)].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def metric_dict(y_true, y_pred, y_score=None) -> Dict[str, float]:
    output = {
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
        output["roc_auc"] = roc_auc_score(y_true, y_score)

    return output


def make_split(df: pd.DataFrame) -> Tuple[pd.Index, pd.Index]:
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=0.2,
        stratify=df[TARGET_COL],
        random_state=RANDOM_STATE,
    )
    return pd.Index(train_idx), pd.Index(test_idx)


def tune_threshold(y_true: pd.Series, y_score: np.ndarray) -> Tuple[float, pd.DataFrame]:
    rows = []
    thresholds = np.arange(0.25, 0.76, 0.01)

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
                "recall_rejected": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
                "recall_accepted": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
                "recall_gap_abs": abs(
                    recall_score(y_true, y_pred, pos_label=0, zero_division=0)
                    - recall_score(y_true, y_pred, pos_label=1, zero_division=0)
                ),
            }
        )

    table = pd.DataFrame(rows)

    table["selection_score"] = table["f1_macro"] - 0.25 * table["recall_gap_abs"]
    best = table.sort_values(["selection_score", "f1_macro"], ascending=False).iloc[0]

    return float(best["threshold"]), table


def train_model(
    df: pd.DataFrame,
    features: Sequence[str],
    model_name: str,
    train_idx: pd.Index,
    test_idx: pd.Index,
    tune_threshold_flag: bool = False,
) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    features = available_columns(df, features, model_name)

    X_train = coerce_features(df.loc[train_idx], features)
    X_test = coerce_features(df.loc[test_idx], features)
    y_train = df.loc[train_idx, TARGET_COL]
    y_test = df.loc[test_idx, TARGET_COL]

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    train_score = pipeline.predict_proba(X_train)[:, 1]
    test_score = pipeline.predict_proba(X_test)[:, 1]

    threshold = 0.50
    threshold_table = pd.DataFrame()

    if tune_threshold_flag:
        threshold, threshold_table = tune_threshold(y_train, train_score)

    y_pred = (test_score >= threshold).astype(int)
    metrics = metric_dict(y_test, y_pred, test_score)

    result = {
        "model_name": model_name,
        "used_features": len(features),
        "threshold": threshold,
        **metrics,
        "train_n": len(train_idx),
        "test_n": len(test_idx),
    }

    predictions = df.loc[test_idx].copy()
    predictions["_model_name"] = model_name
    predictions["_y_true"] = y_test.values
    predictions["_y_score"] = test_score
    predictions["_y_pred"] = y_pred

    return result, predictions, threshold_table


def run_model_comparison(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    print("\nRunning mitigation model comparison")
    print("=" * 80)

    train_idx, test_idx = make_split(df)

    specs = [
        ("Baseline A structural temporal", MODEL_A_FEATURES, False),
        ("Baseline C with party indicators", MODEL_C_FEATURES, False),
        ("Mitigated D with missingness indicators", MODEL_D_MITIGATED_FEATURES, False),
        ("Mitigated D with threshold tuning", MODEL_D_MITIGATED_FEATURES, True),
    ]

    rows = []
    prediction_tables = []
    threshold_tables = []

    for model_name, features, tune_flag in specs:
        print(f"Training {model_name}")
        result, predictions, threshold_table = train_model(
            df=df,
            features=features,
            model_name=model_name,
            train_idx=train_idx,
            test_idx=test_idx,
            tune_threshold_flag=tune_flag,
        )

        rows.append(result)
        prediction_tables.append(predictions)

        if not threshold_table.empty:
            threshold_table["model_name"] = model_name
            threshold_tables.append(threshold_table)

    comparison = pd.DataFrame(rows)
    predictions_all = pd.concat(prediction_tables, ignore_index=True)
    thresholds_all = pd.concat(threshold_tables, ignore_index=True) if threshold_tables else pd.DataFrame()

    save_csv(comparison, "mitigation_model_comparison.csv")
    save_csv(predictions_all[["_model_name", "_y_true", "_y_score", "_y_pred"]], "mitigation_predictions.csv")

    if not thresholds_all.empty:
        save_csv(thresholds_all, "threshold_tuning_results.csv")

    print(comparison.to_string(index=False))

    return {
        "comparison": comparison,
        "predictions": predictions_all,
        "thresholds": thresholds_all,
    }


def class_performance(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for model_name, group in predictions.groupby("_model_name"):
        report = classification_report(
            group["_y_true"],
            group["_y_pred"],
            output_dict=True,
            zero_division=0,
        )

        for class_value, class_label in [("0", "rejected"), ("1", "accepted")]:
            row = report[class_value]
            rows.append(
                {
                    "model_name": model_name,
                    "class_value": class_value,
                    "class_label": class_label,
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1_score": row["f1-score"],
                    "support": row["support"],
                }
            )

    table = pd.DataFrame(rows)
    save_csv(table, "mitigation_class_performance.csv")
    return table


def confusion_matrices(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for model_name, group in predictions.groupby("_model_name"):
        cm = confusion_matrix(group["_y_true"], group["_y_pred"], labels=[0, 1])
        rows.append(
            {
                "model_name": model_name,
                "true_rejected_pred_rejected": int(cm[0, 0]),
                "true_rejected_pred_accepted": int(cm[0, 1]),
                "true_accepted_pred_rejected": int(cm[1, 0]),
                "true_accepted_pred_accepted": int(cm[1, 1]),
            }
        )

    table = pd.DataFrame(rows)
    save_csv(table, "mitigation_confusion_matrices.csv")
    return table


def subgroup_performance(predictions: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "speaker_party_clean",
        "time_bin",
        "is_voorzitter",
        "sentiment_label",
        "long_speech_flag",
        "year",
    ]

    rows = []

    for model_name, model_df in predictions.groupby("_model_name"):
        for group_col in group_cols:
            if group_col not in model_df.columns:
                continue

            work = model_df[[group_col, "_y_true", "_y_pred"]].copy()
            work[group_col] = work[group_col].astype("object").where(work[group_col].notna(), "MISSING_GROUP_VALUE")

            for group_value, group in work.groupby(group_col):
                if len(group) < MIN_GROUP_N:
                    continue

                if group["_y_true"].nunique() < 2:
                    continue

                rows.append(
                    {
                        "model_name": model_name,
                        "group_column": group_col,
                        "group_value": str(group_value),
                        "n": len(group),
                        "accuracy": accuracy_score(group["_y_true"], group["_y_pred"]),
                        "balanced_accuracy": balanced_accuracy_score(group["_y_true"], group["_y_pred"]),
                        "f1_macro": f1_score(group["_y_true"], group["_y_pred"], average="macro", zero_division=0),
                    }
                )

    table = pd.DataFrame(rows)
    save_csv(table, "mitigation_subgroup_performance.csv")
    return table


def compare_before_after(model_results: Dict[str, pd.DataFrame], class_table: pd.DataFrame) -> pd.DataFrame:
    comparison = model_results["comparison"]

    rows = []

    baseline = comparison[comparison["model_name"] == "Baseline C with party indicators"]
    mitigated = comparison[comparison["model_name"] == "Mitigated D with threshold tuning"]

    if not baseline.empty and not mitigated.empty:
        b = baseline.iloc[0]
        m = mitigated.iloc[0]
        rows.append(
            {
                "comparison": "Baseline C versus mitigated D with threshold tuning",
                "baseline_macro_f1": b["f1_macro"],
                "mitigated_macro_f1": m["f1_macro"],
                "macro_f1_change": m["f1_macro"] - b["f1_macro"],
                "baseline_balanced_accuracy": b["balanced_accuracy"],
                "mitigated_balanced_accuracy": m["balanced_accuracy"],
                "balanced_accuracy_change": m["balanced_accuracy"] - b["balanced_accuracy"],
                "baseline_threshold": b["threshold"],
                "mitigated_threshold": m["threshold"],
            }
        )

    class_pivot = class_table.pivot_table(
        index=["model_name"],
        columns="class_label",
        values=["recall", "f1_score"],
        aggfunc="first",
    )

    if "Baseline C with party indicators" in class_pivot.index and "Mitigated D with threshold tuning" in class_pivot.index:
        b = class_pivot.loc["Baseline C with party indicators"]
        m = class_pivot.loc["Mitigated D with threshold tuning"]

        rows.append(
            {
                "comparison": "Outcome class gap before and after mitigation",
                "baseline_macro_f1": np.nan,
                "mitigated_macro_f1": np.nan,
                "macro_f1_change": np.nan,
                "baseline_balanced_accuracy": np.nan,
                "mitigated_balanced_accuracy": np.nan,
                "balanced_accuracy_change": np.nan,
                "baseline_threshold": np.nan,
                "mitigated_threshold": np.nan,
                "baseline_recall_gap_abs": abs(b[("recall", "rejected")] - b[("recall", "accepted")]),
                "mitigated_recall_gap_abs": abs(m[("recall", "rejected")] - m[("recall", "accepted")]),
                "recall_gap_change": abs(m[("recall", "rejected")] - m[("recall", "accepted")])
                - abs(b[("recall", "rejected")] - b[("recall", "accepted")]),
            }
        )

    table = pd.DataFrame(rows)
    save_csv(table, "mitigation_before_after_summary.csv")
    return table


def write_mitigation_summary(
    mitigation_log: pd.DataFrame,
    model_results: Dict[str, pd.DataFrame],
    class_table: pd.DataFrame,
    before_after: pd.DataFrame,
) -> None:
    comparison = model_results["comparison"]

    lines = [
        "Week 8 Bias Mitigation Implementation Summary",
        "=" * 80,
        "",
        "Purpose",
        "",
        "This script implements feasible mitigation steps based on the bias audit. It does not claim to remove all bias. Some risks, especially visibility bias, require additional data that is not available in the public debate records.",
        "",
        "Implemented mitigation steps",
        "",
    ]

    for _, row in mitigation_log.iterrows():
        lines.append(f"* {row['mitigation_step']}: {row['reason']}")

    lines.extend(
        [
            "",
            "Model comparison",
            "",
        ]
    )

    for _, row in comparison.iterrows():
        lines.append(
            f"* {row['model_name']}: macro F1 {row['f1_macro']:.4f}, "
            f"balanced accuracy {row['balanced_accuracy']:.4f}, "
            f"ROC AUC {row['roc_auc']:.4f}, threshold {row['threshold']:.2f}."
        )

    lines.extend(
        [
            "",
            "Outcome class performance",
            "",
        ]
    )

    for _, row in class_table.iterrows():
        lines.append(
            f"* {row['model_name']} for {row['class_label']}: "
            f"recall {row['recall']:.4f}, F1 {row['f1_score']:.4f}."
        )

    lines.extend(
        [
            "",
            "Before and after summary",
            "",
        ]
    )

    if before_after.empty:
        lines.append("* No before and after comparison could be created.")
    else:
        for _, row in before_after.iterrows():
            lines.append(f"* {row['comparison']}")
            for col in before_after.columns:
                if col == "comparison":
                    continue
                value = row[col]
                if pd.notna(value):
                    if isinstance(value, float):
                        lines.append(f"  {col}: {value:.4f}")
                    else:
                        lines.append(f"  {col}: {value}")

    lines.extend(
        [
            "",
            "Interpretation",
            "",
            "The mitigation implementation handles the parts of bias that can be addressed with the available data. Missingness is addressed by adding missingness flags and removing fully empty columns. Omitted variable bias is partly addressed by adding party indicators. Outcome imbalance is addressed by class weighting, class level reporting, and threshold tuning. Feature bias is addressed by treating timing, speech duration, sentiment, and speaker frequency as visible indicators rather than direct causes. Visibility bias cannot be fully solved without additional data on informal negotiation, committee bargaining, lobbying, amendments, or party discipline.",
            "",
        ]
    )

    path = OUTPUT_DIR / "week8_bias_mitigation_summary.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {path}")


def main() -> None:
    print("Week 8 Bias Mitigation Implementation")
    print("=" * 80)

    ensure_output_dir()

    df = load_data()
    mitigated_df = create_mitigated_dataset(df)

    mitigation_log = pd.read_csv(OUTPUT_DIR / "mitigation_steps_log.csv")

    model_results = run_model_comparison(mitigated_df)
    class_table = class_performance(model_results["predictions"])
    confusion_matrices(model_results["predictions"])
    subgroup_performance(model_results["predictions"])

    before_after = compare_before_after(model_results, class_table)

    write_mitigation_summary(
        mitigation_log=mitigation_log,
        model_results=model_results,
        class_table=class_table,
        before_after=before_after,
    )

    print("\nDone.")
    print(f"Outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()