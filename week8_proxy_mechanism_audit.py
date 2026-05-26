"""
Week 8 Proxy Mechanism Audit

Purpose
This script tests whether visible parliamentary debate variables are patterned by
party, role, topic, timing, and outcome. It is meant as an interpretive mitigation
step after the systematic bias audit.

It does three things:
1. Association tests
2. Party level clustering
3. PCA dimension reduction

Run from the repository root with:

    py -3.12 week8_proxy_mechanism_audit.py

Main input:
    parliamentary_notebooks_speeches/speeches_with_sentiment.csv

Optional topic input:
    feature-engineering/engineered_topics2.csv

Outputs:
    week8_proxy_mechanism_outputs
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from scipy.stats import chi2_contingency, kruskal
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


RANDOM_STATE = 42
MIN_GROUP_N = 50
MIN_PARTY_N = 100
MAX_CATEGORY_LEVELS = 40


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def find_repo_root() -> Path:
    expected = Path("parliamentary_notebooks_speeches") / "speeches_with_sentiment.csv"

    candidates: List[Path] = []
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

    seen: List[Path] = []
    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except Exception:
            continue
        if candidate not in seen:
            seen.append(candidate)

    for candidate in seen:
        if (candidate / expected).exists():
            return candidate

    return cwd


REPO_ROOT = find_repo_root()
SPEECH_DATA_PATH = REPO_ROOT / "parliamentary_notebooks_speeches" / "speeches_with_sentiment.csv"
TOPIC_DATA_PATH = REPO_ROOT / "feature-engineering" / "engineered_topics2.csv"
OUTPUT_DIR = REPO_ROOT / "week8_proxy_mechanism_outputs"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

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


def save_text(text: str, filename: str) -> Path:
    ensure_output_dir()
    path = OUTPUT_DIR / filename
    path.write_text(text, encoding="utf-8")
    print(f"Saved {path}")
    return path


def available_columns(df: pd.DataFrame, columns: Sequence[str]) -> List[str]:
    return [col for col in columns if col in df.columns]


def mode_or_missing(series: pd.Series) -> object:
    series = series.dropna()
    if series.empty:
        return np.nan
    return series.mode().iloc[0]


def safe_rate(numerator: float, denominator: float) -> float:
    if denominator == 0 or pd.isna(denominator):
        return np.nan
    return numerator / denominator


def safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return np.nan


def format_rate(value: float) -> str:
    if pd.isna(value):
        return "not available"
    return f"{100 * value:.1f} percent"


def format_number(value: float) -> str:
    if pd.isna(value):
        return "not available"
    return f"{value:.4f}"


# ---------------------------------------------------------------------------
# Load and merge data
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    print("Loading data")
    print("=" * 80)

    if not SPEECH_DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find speech data at {SPEECH_DATA_PATH}")

    speeches = pd.read_csv(SPEECH_DATA_PATH, low_memory=False)
    print(f"Loaded speech data: {SPEECH_DATA_PATH}")
    print(f"Speech rows: {len(speeches):,}")
    print(f"Speech columns: {speeches.shape[1]:,}")

    if "label" in speeches.columns:
        speeches["label"] = pd.to_numeric(speeches["label"], errors="coerce")

    if "speaker_party_clean" not in speeches.columns and "speaker_party" in speeches.columns:
        speeches["speaker_party_clean"] = speeches["speaker_party"].fillna("Missing")

    if TOPIC_DATA_PATH.exists():
        topics = pd.read_csv(TOPIC_DATA_PATH, low_memory=False)
        print(f"Loaded topic data: {TOPIC_DATA_PATH}")
        print(f"Topic rows: {len(topics):,}")

        if "matched_activiteit_id" in speeches.columns and "Activiteit_Id" in topics.columns:
            topic_cols = [
                "Activiteit_Id",
                "Topic_category",
                "Meeting_type",
                "Topic_duration_minutes",
                "Topic_density_per_day",
                "Time_of_day_category",
                "Season",
            ]
            topic_cols = available_columns(topics, topic_cols)
            topics_small = topics[topic_cols].drop_duplicates(subset=["Activiteit_Id"])
            before_cols = speeches.shape[1]
            speeches = speeches.merge(
                topics_small,
                left_on="matched_activiteit_id",
                right_on="Activiteit_Id",
                how="left",
            )
            print(f"Merged topic features. Columns before: {before_cols}, after: {speeches.shape[1]}")
        else:
            warn("Could not merge topic data because merge keys were unavailable.")
    else:
        warn(f"Topic file not found, continuing without topic merge: {TOPIC_DATA_PATH}")

    return speeches


# ---------------------------------------------------------------------------
# Association tests
# ---------------------------------------------------------------------------

def eta_squared_from_groups(df: pd.DataFrame, group_col: str, value_col: str) -> float:
    work = df[[group_col, value_col]].dropna().copy()
    if work.empty:
        return np.nan

    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna()
    if work.empty:
        return np.nan

    overall_mean = work[value_col].mean()
    total_ss = ((work[value_col] - overall_mean) ** 2).sum()
    if total_ss == 0:
        return 0.0

    between_ss = 0.0
    for _, group_df in work.groupby(group_col):
        group_mean = group_df[value_col].mean()
        between_ss += len(group_df) * ((group_mean - overall_mean) ** 2)

    return between_ss / total_ss


def cramers_v_from_table(table: pd.DataFrame) -> Tuple[float, float]:
    if table.empty:
        return np.nan, np.nan

    observed = table.values
    n = observed.sum()
    if n == 0:
        return np.nan, np.nan

    if SCIPY_AVAILABLE:
        chi2, p_value, _, _ = chi2_contingency(observed)
    else:
        chi2, p_value = np.nan, np.nan

    r, k = observed.shape
    denominator = n * (min(k - 1, r - 1))
    if denominator == 0 or pd.isna(chi2):
        return np.nan, p_value

    return float(np.sqrt(chi2 / denominator)), float(p_value)


def numeric_association_test(df: pd.DataFrame, group_col: str, value_col: str) -> Dict[str, object]:
    work = df[[group_col, value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna()

    if work.empty:
        return {}

    counts = work[group_col].value_counts()
    keep = counts[counts >= MIN_GROUP_N].index
    work = work[work[group_col].isin(keep)].copy()

    if work[group_col].nunique() < 2:
        return {}

    group_summary = (
        work.groupby(group_col)[value_col]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    group_summary.to_csv(OUTPUT_DIR / f"group_summary_{group_col}_by_{value_col}.csv", index=False)

    eta_sq = eta_squared_from_groups(work, group_col, value_col)

    p_value = np.nan
    if SCIPY_AVAILABLE:
        groups = [group[value_col].values for _, group in work.groupby(group_col)]
        if len(groups) >= 2:
            try:
                _, p_value = kruskal(*groups)
            except Exception:
                p_value = np.nan

    top = group_summary.iloc[0]
    bottom = group_summary.iloc[-1]

    return {
        "test_type": "numeric_group_association",
        "question": f"Does {group_col} structure {value_col}?",
        "group_column": group_col,
        "value_column": value_col,
        "n_rows_used": len(work),
        "n_groups_used": work[group_col].nunique(),
        "effect_size": eta_sq,
        "effect_size_name": "eta_squared",
        "p_value": p_value,
        "highest_group": str(top[group_col]),
        "highest_group_mean": top["mean"],
        "lowest_group": str(bottom[group_col]),
        "lowest_group_mean": bottom["mean"],
        "interpretation": "Higher eta squared means stronger group patterning of the numeric variable.",
    }


def categorical_association_test(df: pd.DataFrame, row_col: str, col_col: str) -> Dict[str, object]:
    work = df[[row_col, col_col]].dropna().copy()
    if work.empty:
        return {}

    row_counts = work[row_col].value_counts()
    col_counts = work[col_col].value_counts()
    keep_rows = row_counts[row_counts >= MIN_GROUP_N].index
    keep_cols = col_counts[col_counts >= MIN_GROUP_N].index
    work = work[work[row_col].isin(keep_rows) & work[col_col].isin(keep_cols)].copy()

    if work[row_col].nunique() < 2 or work[col_col].nunique() < 2:
        return {}

    table = pd.crosstab(work[row_col], work[col_col])
    proportions = pd.crosstab(work[row_col], work[col_col], normalize="index")

    table.to_csv(OUTPUT_DIR / f"crosstab_{row_col}_by_{col_col}.csv")
    proportions.to_csv(OUTPUT_DIR / f"crosstab_share_{row_col}_by_{col_col}.csv")

    v, p_value = cramers_v_from_table(table)

    strongest_cells = []
    for idx in proportions.index:
        top_col = proportions.loc[idx].idxmax()
        strongest_cells.append(
            {
                row_col: idx,
                "most_common_category": top_col,
                "share": proportions.loc[idx, top_col],
            }
        )
    strongest_df = pd.DataFrame(strongest_cells)
    strongest_df.to_csv(OUTPUT_DIR / f"dominant_category_{row_col}_by_{col_col}.csv", index=False)

    return {
        "test_type": "categorical_association",
        "question": f"Is {row_col} associated with {col_col}?",
        "group_column": row_col,
        "value_column": col_col,
        "n_rows_used": len(work),
        "n_groups_used": work[row_col].nunique(),
        "effect_size": v,
        "effect_size_name": "cramers_v",
        "p_value": p_value,
        "highest_group": "see dominant category output",
        "highest_group_mean": np.nan,
        "lowest_group": "see dominant category output",
        "lowest_group_mean": np.nan,
        "interpretation": "Higher Cramers V means stronger association between the two categorical variables.",
    }


def run_association_tests(df: pd.DataFrame) -> pd.DataFrame:
    print("\nRunning association tests")
    print("=" * 80)

    ensure_output_dir()
    rows: List[Dict[str, object]] = []

    numeric_tests = [
        ("speaker_party_clean", "speech_duration_seconds"),
        ("speaker_party_clean", "log_speech_duration"),
        ("speaker_party_clean", "speaker_freq"),
        ("speaker_party_clean", "log_speaker_freq"),
        ("speaker_party_clean", "sentiment_score"),
        ("speaker_party_clean", "sentiment_pos"),
        ("speaker_party_clean", "sentiment_neg"),
        ("speaker_role", "speech_duration_seconds"),
        ("speaker_role", "sentiment_score"),
        ("time_bin", "speech_duration_seconds"),
        ("time_bin", "sentiment_score"),
        ("Topic_category", "speech_duration_seconds"),
        ("Topic_category", "sentiment_score"),
        ("Meeting_type", "speech_duration_seconds"),
        ("Meeting_type", "sentiment_score"),
    ]

    categorical_tests = [
        ("speaker_party_clean", "time_bin"),
        ("speaker_party_clean", "sentiment_label"),
        ("speaker_party_clean", "Topic_category"),
        ("speaker_party_clean", "Meeting_type"),
        ("speaker_party_clean", "Time_of_day_category"),
        ("speaker_party_clean", "motion_outcome_raw"),
        ("speaker_role", "time_bin"),
        ("speaker_role", "sentiment_label"),
        ("time_bin", "motion_outcome_raw"),
        ("Topic_category", "motion_outcome_raw"),
    ]

    for group_col, value_col in numeric_tests:
        if group_col in df.columns and value_col in df.columns:
            result = numeric_association_test(df, group_col, value_col)
            if result:
                rows.append(result)
        else:
            warn(f"Skipping numeric test because columns are unavailable: {group_col}, {value_col}")

    for row_col, col_col in categorical_tests:
        if row_col in df.columns and col_col in df.columns:
            result = categorical_association_test(df, row_col, col_col)
            if result:
                rows.append(result)
        else:
            warn(f"Skipping categorical test because columns are unavailable: {row_col}, {col_col}")

    results = pd.DataFrame(rows)
    save_csv(results, "association_tests.csv")

    if results.empty:
        print("No association tests completed.")
    else:
        print(results.sort_values("effect_size", ascending=False).head(15).to_string(index=False))

    return results


# ---------------------------------------------------------------------------
# Party profile table
# ---------------------------------------------------------------------------

def make_party_profile_table(df: pd.DataFrame) -> pd.DataFrame:
    print("\nCreating party profile table")
    print("=" * 80)

    if "speaker_party_clean" not in df.columns:
        raise ValueError("speaker_party_clean is required for party profiles.")

    work = df[df["speaker_party_clean"].notna()].copy()
    party_counts = work["speaker_party_clean"].value_counts()
    keep_parties = party_counts[party_counts >= MIN_PARTY_N].index
    work = work[work["speaker_party_clean"].isin(keep_parties)].copy()

    rows = []
    for party, group in work.groupby("speaker_party_clean"):
        n = len(group)

        row = {
            "speaker_party_clean": party,
            "n_speeches": n,
            "n_unique_speakers": group["speaker_id"].nunique() if "speaker_id" in group.columns else np.nan,
            "n_unique_motions": group["motion_id"].nunique() if "motion_id" in group.columns else np.nan,
            "mean_speech_duration": pd.to_numeric(group.get("speech_duration_seconds"), errors="coerce").mean() if "speech_duration_seconds" in group.columns else np.nan,
            "median_speech_duration": pd.to_numeric(group.get("speech_duration_seconds"), errors="coerce").median() if "speech_duration_seconds" in group.columns else np.nan,
            "total_speech_duration": pd.to_numeric(group.get("speech_duration_seconds"), errors="coerce").sum() if "speech_duration_seconds" in group.columns else np.nan,
            "mean_log_speech_duration": pd.to_numeric(group.get("log_speech_duration"), errors="coerce").mean() if "log_speech_duration" in group.columns else np.nan,
            "mean_speaker_freq": pd.to_numeric(group.get("speaker_freq"), errors="coerce").mean() if "speaker_freq" in group.columns else np.nan,
            "mean_log_speaker_freq": pd.to_numeric(group.get("log_speaker_freq"), errors="coerce").mean() if "log_speaker_freq" in group.columns else np.nan,
            "mean_sentiment_score": pd.to_numeric(group.get("sentiment_score"), errors="coerce").mean() if "sentiment_score" in group.columns else np.nan,
            "mean_sentiment_pos": pd.to_numeric(group.get("sentiment_pos"), errors="coerce").mean() if "sentiment_pos" in group.columns else np.nan,
            "mean_sentiment_neg": pd.to_numeric(group.get("sentiment_neg"), errors="coerce").mean() if "sentiment_neg" in group.columns else np.nan,
            "share_negative_sentiment": safe_rate((group.get("sentiment_label") == "negative").sum(), n) if "sentiment_label" in group.columns else np.nan,
            "share_positive_sentiment": safe_rate((group.get("sentiment_label") == "positive").sum(), n) if "sentiment_label" in group.columns else np.nan,
            "share_chair_speeches": pd.to_numeric(group.get("is_voorzitter"), errors="coerce").mean() if "is_voorzitter" in group.columns else np.nan,
            "share_long_speeches": pd.to_numeric(group.get("long_speech_flag"), errors="coerce").mean() if "long_speech_flag" in group.columns else np.nan,
            "share_morning": safe_rate((group.get("time_bin") == "morning").sum(), n) if "time_bin" in group.columns else np.nan,
            "share_afternoon": safe_rate((group.get("time_bin") == "afternoon").sum(), n) if "time_bin" in group.columns else np.nan,
            "share_evening": safe_rate((group.get("time_bin") == "evening").sum(), n) if "time_bin" in group.columns else np.nan,
            "accepted_rate": pd.to_numeric(group.get("label"), errors="coerce").mean() if "label" in group.columns else np.nan,
            "dominant_topic_category": mode_or_missing(group["Topic_category"]) if "Topic_category" in group.columns else np.nan,
            "dominant_meeting_type": mode_or_missing(group["Meeting_type"]) if "Meeting_type" in group.columns else np.nan,
            "dominant_time_category": mode_or_missing(group["Time_of_day_category"]) if "Time_of_day_category" in group.columns else np.nan,
        }

        if "Topic_category" in group.columns:
            topic_counts = group["Topic_category"].value_counts(normalize=True)
            for topic, share in topic_counts.items():
                clean_topic = str(topic).lower().replace(" ", "_").replace("&", "and").replace("/", "_")
                row[f"topic_share_{clean_topic}"] = share

        rows.append(row)

    profile = pd.DataFrame(rows)
    profile = profile.sort_values("n_speeches", ascending=False)
    save_csv(profile, "party_profile_table.csv")
    print(profile.head(20).to_string(index=False))
    return profile


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def choose_cluster_features(profile: pd.DataFrame) -> List[str]:
    base_features = [
        "n_speeches",
        "n_unique_speakers",
        "n_unique_motions",
        "mean_speech_duration",
        "median_speech_duration",
        "mean_log_speech_duration",
        "mean_speaker_freq",
        "mean_log_speaker_freq",
        "mean_sentiment_score",
        "mean_sentiment_pos",
        "mean_sentiment_neg",
        "share_negative_sentiment",
        "share_chair_speeches",
        "share_long_speeches",
        "share_morning",
        "share_afternoon",
        "share_evening",
        "accepted_rate",
    ]

    topic_features = [col for col in profile.columns if col.startswith("topic_share_")]
    features = available_columns(profile, base_features + topic_features)

    usable = []
    for col in features:
        numeric = pd.to_numeric(profile[col], errors="coerce")
        if numeric.notna().sum() >= 3 and numeric.nunique(dropna=True) > 1:
            usable.append(col)

    return usable


def run_party_clustering(profile: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("\nRunning party clustering")
    print("=" * 80)

    features = choose_cluster_features(profile)
    if len(features) < 2 or len(profile) < 4:
        warn("Not enough party profile features or parties for clustering.")
        empty = pd.DataFrame()
        save_csv(empty, "party_clusters.csv")
        save_csv(empty, "cluster_quality.csv")
        return empty, empty

    x = profile[features].copy()
    for col in features:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    preprocessing = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    x_scaled = preprocessing.fit_transform(x)

    quality_rows = []
    max_k = min(6, len(profile) - 1)
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = model.fit_predict(x_scaled)
        try:
            score = silhouette_score(x_scaled, labels)
        except Exception:
            score = np.nan
        quality_rows.append({"k": k, "silhouette_score": score})

    quality = pd.DataFrame(quality_rows)
    save_csv(quality, "cluster_quality.csv")

    if quality.empty or quality["silhouette_score"].isna().all():
        chosen_k = 3 if len(profile) >= 3 else 2
    else:
        chosen_k = int(quality.sort_values("silhouette_score", ascending=False).iloc[0]["k"])

    cluster_model = KMeans(n_clusters=chosen_k, random_state=RANDOM_STATE, n_init=20)
    labels = cluster_model.fit_predict(x_scaled)

    clustered = profile.copy()
    clustered["cluster"] = labels
    clustered["cluster_k"] = chosen_k

    save_csv(clustered, "party_clusters.csv")

    cluster_summary = (
        clustered.groupby("cluster")
        .agg(
            n_parties=("speaker_party_clean", "count"),
            parties=("speaker_party_clean", lambda s: ", ".join(map(str, s))),
            mean_n_speeches=("n_speeches", "mean"),
            mean_speech_duration=("mean_speech_duration", "mean"),
            mean_speaker_freq=("mean_speaker_freq", "mean"),
            mean_sentiment_score=("mean_sentiment_score", "mean"),
            mean_share_evening=("share_evening", "mean"),
            mean_share_long_speeches=("share_long_speeches", "mean"),
            mean_accepted_rate=("accepted_rate", "mean"),
        )
        .reset_index()
    )
    save_csv(cluster_summary, "cluster_summary.csv")

    print("Cluster quality:")
    print(quality.to_string(index=False))
    print("Cluster summary:")
    print(cluster_summary.to_string(index=False))

    return clustered, quality


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def run_party_pca(profile: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("\nRunning PCA on party profiles")
    print("=" * 80)

    features = choose_cluster_features(profile)
    if len(features) < 2 or len(profile) < 3:
        warn("Not enough party profile features or parties for PCA.")
        empty = pd.DataFrame()
        save_csv(empty, "pca_party_dimensions.csv")
        save_csv(empty, "pca_feature_loadings.csv")
        save_csv(empty, "pca_explained_variance.csv")
        return empty, empty, empty

    x = profile[features].copy()
    for col in features:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    preprocessing = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    x_scaled = preprocessing.fit_transform(x)

    n_components = min(3, x_scaled.shape[0], x_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    pcs = pca.fit_transform(x_scaled)

    pca_table = profile[["speaker_party_clean", "n_speeches"]].copy()
    for i in range(n_components):
        pca_table[f"pc{i + 1}"] = pcs[:, i]

    explained = pd.DataFrame(
        {
            "component": [f"pc{i + 1}" for i in range(n_components)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )

    loadings_rows = []
    for component_idx in range(n_components):
        for feature_idx, feature in enumerate(features):
            loadings_rows.append(
                {
                    "component": f"pc{component_idx + 1}",
                    "feature": feature,
                    "loading": pca.components_[component_idx, feature_idx],
                    "abs_loading": abs(pca.components_[component_idx, feature_idx]),
                }
            )

    loadings = pd.DataFrame(loadings_rows)
    loadings = loadings.sort_values(["component", "abs_loading"], ascending=[True, False])

    save_csv(pca_table, "pca_party_dimensions.csv")
    save_csv(loadings, "pca_feature_loadings.csv")
    save_csv(explained, "pca_explained_variance.csv")

    print("PCA explained variance:")
    print(explained.to_string(index=False))
    print("Top loadings:")
    print(loadings.groupby("component").head(8).to_string(index=False))

    return pca_table, loadings, explained


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize_top_associations(association_results: pd.DataFrame) -> List[str]:
    if association_results.empty:
        return ["No association tests completed successfully."]

    lines = []
    top = association_results.sort_values("effect_size", ascending=False).head(10)
    for _, row in top.iterrows():
        lines.append(
            f"* {row['question']}: {row['effect_size_name']} = {format_number(row['effect_size'])}. "
            f"Highest group: {row['highest_group']}. Lowest group: {row['lowest_group']}."
        )
    return lines


def summarize_clusters(clustered: pd.DataFrame) -> List[str]:
    if clustered.empty or "cluster" not in clustered.columns:
        return ["No party clusters were created."]

    lines = []
    for cluster, group in clustered.groupby("cluster"):
        parties = ", ".join(group["speaker_party_clean"].astype(str).tolist())
        lines.append(f"* Cluster {cluster}: {parties}")
    return lines


def summarize_pca(loadings: pd.DataFrame, explained: pd.DataFrame) -> List[str]:
    if loadings.empty or explained.empty:
        return ["No PCA dimensions were created."]

    lines = []
    for component, group in loadings.groupby("component"):
        top_features = group.sort_values("abs_loading", ascending=False).head(5)
        explained_row = explained[explained["component"] == component]
        explained_share = explained_row.iloc[0]["explained_variance_ratio"] if not explained_row.empty else np.nan
        feature_text = ", ".join(
            f"{row['feature']} ({row['loading']:.2f})" for _, row in top_features.iterrows()
        )
        lines.append(
            f"* {component} explains {format_rate(explained_share)} of profile variance. "
            f"Largest loadings: {feature_text}."
        )
    return lines


def write_summary(
    association_results: pd.DataFrame,
    profile: pd.DataFrame,
    clustered: pd.DataFrame,
    cluster_quality: pd.DataFrame,
    pca_table: pd.DataFrame,
    loadings: pd.DataFrame,
    explained: pd.DataFrame,
) -> None:
    print("\nWriting summary")
    print("=" * 80)

    lines = [
        "Week 8 Proxy Mechanism Audit Summary",
        "=" * 80,
        "",
        "Purpose",
        "",
        "This audit tests whether visible parliamentary variables are systematically patterned by party, role, topic, timing, and outcome. It does not claim causality. Instead, it checks whether variables such as speech duration, speaker frequency, sentiment, and time of day can reasonably be interpreted as indicators of deeper political or institutional mechanisms.",
        "",
        "Methods used",
        "",
        "* Association tests: numerical group differences and categorical cross tabs.",
        "* Clustering: party level profiles grouped into behavioural clusters.",
        "* PCA: party level profile variables reduced into broader dimensions.",
        "",
        "Association test highlights",
        "",
    ]

    lines.extend(summarize_top_associations(association_results))

    lines.extend(
        [
            "",
            "Party profile table",
            "",
            f"The party profile table includes {len(profile)} parties with at least {MIN_PARTY_N} speeches. It summarises speech volume, speech duration, speaker frequency, sentiment, time of day, chair share, long speech share, accepted rate, and topic patterns.",
            "",
            "Clustering results",
            "",
        ]
    )

    lines.extend(summarize_clusters(clustered))

    if not cluster_quality.empty:
        best = cluster_quality.sort_values("silhouette_score", ascending=False).iloc[0]
        lines.append("")
        lines.append(
            f"The selected number of clusters was based on silhouette score. Best k was {int(best['k'])} with score {format_number(best['silhouette_score'])}."
        )

    lines.extend(
        [
            "",
            "PCA results",
            "",
        ]
    )

    lines.extend(summarize_pca(loadings, explained))

    lines.extend(
        [
            "",
            "Interpretation",
            "",
            "The results should be read as evidence of association, not proof of causal power relations. If speech duration, frequency, sentiment, topic concentration, and time of day are patterned by party or role, then these variables should not be interpreted as neutral measures of deliberation. They are better treated as visible indicators of agenda access, political visibility, scheduling patterns, and institutional context.",
            "",
            "Connection to mitigation",
            "",
            "This mitigates the interpretation risk identified in the bias audit. Instead of claiming that visible debate variables directly cause parliamentary outcomes, the project can frame them as observable traces of deeper political mechanisms. Some mechanisms, such as public performance through media presence or audience size, cannot be tested with the current dataset and should be acknowledged as unobserved.",
            "",
        ]
    )

    save_text("\n".join(lines), "proxy_mechanism_summary.txt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Week 8 Proxy Mechanism Audit")
    print("=" * 80)
    print("This script runs association tests, clustering, and PCA.")

    ensure_output_dir()
    df = load_data()

    association_results = run_association_tests(df)
    profile = make_party_profile_table(df)
    clustered, cluster_quality = run_party_clustering(profile)
    pca_table, loadings, explained = run_party_pca(profile)

    write_summary(
        association_results=association_results,
        profile=profile,
        clustered=clustered,
        cluster_quality=cluster_quality,
        pca_table=pca_table,
        loadings=loadings,
        explained=explained,
    )

    print("\nDone.")
    print(f"Outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
