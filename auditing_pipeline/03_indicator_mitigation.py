"""
Week 8 Indicator Mitigation Model

Purpose:
This script tests whether the original visible debate variables are better
understood as indicators of deeper political or institutional mechanisms,
rather than direct predictors of parliamentary outcomes.

It creates three outputs:

03_indicator_mitigation_outputs/indicator_model_results.csv
03_indicator_mitigation_outputs/controlled_outcome_model_results.csv
03_indicator_mitigation_outputs/proxy_interpretation_summary.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
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
MIN_CLASS_COUNT = 100


def find_repo_root() -> Path:
    expected = Path("mindless_machine_pipeline/data") / "speeches_with_sentiment.csv"

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
            home / "Downloads" / "tweedekamer",
        ]
    )

    seen = []
    for candidate in candidates:
        try:
            candidate = candidate.resolve()
            if candidate not in seen:
                seen.append(candidate)
        except Exception:
            pass

    for candidate in seen:
        if (candidate / expected).exists():
            return candidate

    return cwd


REPO_ROOT = find_repo_root()
DATA_PATH = REPO_ROOT / "mindless_machine_pipeline/data" / "speeches_with_sentiment.csv"
OUTPUT_DIR = REPO_ROOT / "03_indicator_mitigation_outputs"


VISIBLE_DEBATE_FEATURES = [
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
    "log_speech_duration",
    "log_text_length",
    "long_speech_flag",
    "log_speaker_freq",
    "sentiment_score",
    "sentiment_pos",
    "sentiment_neg",
]

INSTITUTIONAL_FEATURES = [
    "is_voorzitter",
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

INDICATOR_TARGETS = [
    "is_voorzitter",
    "speaker_party_clean",
    "time_bin",
    "long_speech_flag",
    "sentiment_label",
]


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, filename: str) -> Path:
    ensure_output_dir()
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    print(f"Saved {path}")
    return path


def warn(message: str) -> None:
    print(f"WARNING: {message}")
    warnings.warn(message, stacklevel=2)


def available_columns(df: pd.DataFrame, columns: Sequence[str]) -> List[str]:
    return [col for col in columns if col in df.columns]


def load_data() -> pd.DataFrame:
    print("Loading data")
    print("=" * 80)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find dataset at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, low_memory=False)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df[df[TARGET_COL].isin([0, 1])].copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    print(f"Repository root: {REPO_ROOT}")
    print(f"Loaded: {DATA_PATH}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.shape[1]:,}")
    print("Outcome distribution:")
    print(df[TARGET_COL].value_counts().sort_index())

    return df


def prepare_x(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    available = available_columns(df, features)
    if not available:
        raise ValueError("No available features for model.")

    x = df[available].copy()
    for col in x.columns:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    return x


def logistic_pipeline() -> Pipeline:
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


def random_forest_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    min_samples_leaf=5,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def hist_gradient_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    random_state=RANDOM_STATE,
                    max_iter=200,
                    learning_rate=0.05,
                    max_leaf_nodes=31,
                ),
            ),
        ]
    )


def safe_roc_auc(y_true: pd.Series, y_score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) == 2:
            return roc_auc_score(y_true, y_score)
    except Exception:
        pass
    return np.nan


def metric_row(
    model_name: str,
    target_name: str,
    feature_set_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
    used_features: int,
    train_n: int,
    test_n: int,
) -> Dict[str, object]:
    return {
        "model_name": model_name,
        "target_name": target_name,
        "feature_set": feature_set_name,
        "used_features": used_features,
        "train_n": train_n,
        "test_n": test_n,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "roc_auc": safe_roc_auc(y_true, y_score) if y_score is not None else np.nan,
    }


def get_binary_score(model: Pipeline, x_test: pd.DataFrame) -> np.ndarray | None:
    try:
        return model.predict_proba(x_test)[:, 1]
    except Exception:
        return None


def get_multiclass_score(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> np.ndarray | None:
    try:
        proba = model.predict_proba(x_test)
        if proba.shape[1] == 2:
            return proba[:, 1]
    except Exception:
        pass
    return None


def make_clean_target_frame(
    df: pd.DataFrame,
    target_col: str,
    features: Sequence[str],
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    if target_col not in df.columns:
        raise ValueError(f"Target column not available: {target_col}")

    used_features = available_columns(df, features)
    work = df[used_features + [target_col]].copy()
    work = work[work[target_col].notna()].copy()

    if target_col == TARGET_COL or set(work[target_col].dropna().unique()).issubset({0, 1, True, False}):
        work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
        work = work[work[target_col].isin([0, 1])].copy()
        work[target_col] = work[target_col].astype(int)
    else:
        counts = work[target_col].value_counts(dropna=True)
        keep_values = counts[counts >= MIN_CLASS_COUNT].index
        work = work[work[target_col].isin(keep_values)].copy()
        work[target_col] = work[target_col].astype(str)

    if work[target_col].nunique() < 2:
        raise ValueError(f"Target {target_col} has fewer than two usable classes.")

    x = prepare_x(work, used_features)
    y = work[target_col]

    return x, y, used_features


def run_indicator_models(df: pd.DataFrame) -> pd.DataFrame:
    print("\nRunning indicator models")
    print("=" * 80)

    rows = []
    feature_set_name = "visible debate indicators"

    for target_col in INDICATOR_TARGETS:
        if target_col not in df.columns:
            warn(f"Skipping indicator target because it is missing: {target_col}")
            continue

        try:
            x, y, used_features = make_clean_target_frame(df, target_col, VISIBLE_DEBATE_FEATURES)
        except Exception as exc:
            warn(f"Skipping {target_col}: {exc}")
            continue

        stratify = y if y.value_counts().min() >= 2 else None

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=stratify,
        )

        model_specs = [
            ("Logistic Regression", logistic_pipeline()),
            ("Random Forest", random_forest_pipeline()),
        ]

        if y.nunique() == 2:
            model_specs.append(("Hist Gradient Boosting", hist_gradient_pipeline()))

        for model_name, model in model_specs:
            print(f"Indicator target: {target_col}, model: {model_name}")

            try:
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)

                y_score = None
                if y.nunique() == 2:
                    y_score = get_binary_score(model, x_test)

                rows.append(
                    metric_row(
                        model_name=model_name,
                        target_name=target_col,
                        feature_set_name=feature_set_name,
                        y_true=y_test,
                        y_pred=y_pred,
                        y_score=y_score,
                        used_features=len(used_features),
                        train_n=len(x_train),
                        test_n=len(x_test),
                    )
                )

            except Exception as exc:
                warn(f"Model failed for target {target_col}, model {model_name}: {exc}")

    results = pd.DataFrame(rows)
    save_csv(results, "indicator_model_results.csv")

    print("\nIndicator model results")
    if results.empty:
        print("No indicator models completed.")
    else:
        print(results.sort_values(["target_name", "f1_macro"], ascending=[True, False]).to_string(index=False))

    return results


def run_controlled_outcome_models(df: pd.DataFrame) -> pd.DataFrame:
    print("\nRunning controlled outcome models")
    print("=" * 80)

    feature_sets = {
        "visible debate indicators only": VISIBLE_DEBATE_FEATURES,
        "institutional controls only": INSTITUTIONAL_FEATURES,
        "visible plus institutional controls": VISIBLE_DEBATE_FEATURES + INSTITUTIONAL_FEATURES,
    }

    rows = []

    for feature_set_name, features in feature_sets.items():
        try:
            x, y, used_features = make_clean_target_frame(df, TARGET_COL, features)
        except Exception as exc:
            warn(f"Skipping outcome feature set {feature_set_name}: {exc}")
            continue

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        model_specs = [
            ("Logistic Regression", logistic_pipeline()),
            ("Random Forest", random_forest_pipeline()),
            ("Hist Gradient Boosting", hist_gradient_pipeline()),
        ]

        for model_name, model in model_specs:
            print(f"Outcome model: {feature_set_name}, model: {model_name}")

            try:
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                y_score = get_binary_score(model, x_test)

                rows.append(
                    metric_row(
                        model_name=model_name,
                        target_name=TARGET_COL,
                        feature_set_name=feature_set_name,
                        y_true=y_test,
                        y_pred=y_pred,
                        y_score=y_score,
                        used_features=len(used_features),
                        train_n=len(x_train),
                        test_n=len(x_test),
                    )
                )

            except Exception as exc:
                warn(f"Outcome model failed for {feature_set_name}, {model_name}: {exc}")

    results = pd.DataFrame(rows)
    save_csv(results, "controlled_outcome_model_results.csv")

    print("\nControlled outcome model results")
    if results.empty:
        print("No controlled outcome models completed.")
    else:
        print(results.sort_values(["model_name", "f1_macro"], ascending=[True, False]).to_string(index=False))

    return results


def create_proxy_interpretation(
    indicator_results: pd.DataFrame,
    controlled_results: pd.DataFrame,
) -> None:
    print("\nWriting proxy interpretation summary")
    print("=" * 80)

    lines = [
        "Week 8 Indicator Mitigation Summary",
        "=" * 80,
        "",
        "Purpose",
        "",
        "This model tests the mitigation idea that visible debate variables should not be treated only as direct predictors of parliamentary outcomes. Instead, they may work as indicators of deeper political or institutional mechanisms.",
        "",
        "Indicator model logic",
        "",
        "The indicator models use visible debate features to predict institutional or contextual variables such as chair status, party, time bin, long speech flag, and sentiment label.",
        "",
    ]

    if indicator_results.empty:
        lines.append("No indicator models completed successfully.")
    else:
        best_indicator = (
            indicator_results.sort_values("f1_macro", ascending=False)
            .groupby("target_name")
            .head(1)
            .sort_values("f1_macro", ascending=False)
        )

        lines.append("Best indicator model per target:")
        for _, row in best_indicator.iterrows():
            lines.append(
                f"* {row['target_name']}: best model was {row['model_name']} with "
                f"macro F1 {row['f1_macro']:.4f}, balanced accuracy {row['balanced_accuracy']:.4f}."
            )

    lines.extend(
        [
            "",
            "Controlled outcome model logic",
            "",
            "The controlled outcome models compare visible debate indicators only, institutional controls only, and visible indicators plus institutional controls. If the combined model only slightly improves over institutional controls, this suggests that visible variables are partly acting as proxies for institutional context.",
            "",
        ]
    )

    if controlled_results.empty:
        lines.append("No controlled outcome models completed successfully.")
    else:
        best_outcome = controlled_results.sort_values("f1_macro", ascending=False)
        lines.append("Outcome model results:")
        for _, row in best_outcome.iterrows():
            lines.append(
                f"* {row['model_name']} with {row['feature_set']}: "
                f"macro F1 {row['f1_macro']:.4f}, balanced accuracy {row['balanced_accuracy']:.4f}, "
                f"ROC AUC {row['roc_auc']:.4f}."
            )

        logistic = controlled_results[controlled_results["model_name"] == "Logistic Regression"].copy()
        if not logistic.empty:
            pivot = logistic.pivot_table(
                index="feature_set",
                values="f1_macro",
                aggfunc="first",
            )

            visible_name = "visible debate indicators only"
            inst_name = "institutional controls only"
            combined_name = "visible plus institutional controls"

            if (
                visible_name in pivot.index
                and inst_name in pivot.index
                and combined_name in pivot.index
            ):
                visible_f1 = float(pivot.loc[visible_name, "f1_macro"])
                inst_f1 = float(pivot.loc[inst_name, "f1_macro"])
                combined_f1 = float(pivot.loc[combined_name, "f1_macro"])

                lines.extend(
                    [
                        "",
                        "Logistic regression comparison:",
                        f"* Visible only macro F1: {visible_f1:.4f}",
                        f"* Institutional only macro F1: {inst_f1:.4f}",
                        f"* Visible plus institutional macro F1: {combined_f1:.4f}",
                        f"* Gain from adding institutional controls to visible features: {combined_f1 - visible_f1:.4f}",
                        f"* Gain from adding visible features to institutional controls: {combined_f1 - inst_f1:.4f}",
                    ]
                )

    lines.extend(
        [
            "",
            "Interpretation for mitigation",
            "",
            "This does not remove all bias. Instead, it mitigates the interpretation risk by testing whether visible variables operate as proxies for deeper political mechanisms. If visible variables predict institutional context, and if institutional controls explain outcome patterns as well as or better than visible variables alone, we should avoid claiming that speech duration, sentiment, or timing directly cause outcomes.",
            "",
            "Recommended wording",
            "",
            "The model should be interpreted as an audit of visible parliamentary indicators. These indicators may reflect agenda access, political visibility, scheduling constraints, public performance, and institutional role. They should not be treated as direct measures of deliberative quality.",
            "",
        ]
    )

    path = OUTPUT_DIR / "proxy_interpretation_summary.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {path}")


def create_method_table() -> pd.DataFrame:
    rows = [
        {
            "mitigation_question": "Do visible debate variables directly predict outcomes?",
            "coded_test": "Outcome model with visible debate indicators only",
            "output_file": "controlled_outcome_model_results.csv",
        },
        {
            "mitigation_question": "Do visible debate variables indicate deeper context?",
            "coded_test": "Indicator models predicting chair status, party, time bin, long speech flag, and sentiment label",
            "output_file": "indicator_model_results.csv",
        },
        {
            "mitigation_question": "Do visible variables still add value after institutional controls?",
            "coded_test": "Controlled outcome models comparing visible only, institutional only, and combined feature sets",
            "output_file": "controlled_outcome_model_results.csv",
        },
        {
            "mitigation_question": "How should the results be interpreted?",
            "coded_test": "Proxy interpretation summary",
            "output_file": "proxy_interpretation_summary.txt",
        },
    ]

    table = pd.DataFrame(rows)
    save_csv(table, "indicator_mitigation_method_table.csv")
    return table


def main() -> None:
    print("Week 8 Indicator Mitigation Model")
    print("=" * 80)

    ensure_output_dir()
    df = load_data()

    indicator_results = run_indicator_models(df)
    controlled_results = run_controlled_outcome_models(df)
    create_method_table()
    create_proxy_interpretation(indicator_results, controlled_results)

    print("\nDone.")
    print(f"Outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()