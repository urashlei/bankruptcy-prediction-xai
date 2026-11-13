
from pathlib import Path
from itertools import combinations
import re

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from config import ROOT_DIR, SHAP_DIR, STABILITY_DIR, MODEL_NAMES


# =========================================================
# CONFIGURATION
# =========================================================

TOP_K = 10

FEATURE_COLUMNS = [
    "feature",
    "Feature",
    "attribute",
    "Attribute",
    "variable",
    "Variable",
]

IMPORTANCE_COLUMNS = [
    "mean_abs_shap",
    "mean_abs_SHAP",
    "mean_absolute_shap",
    "mean_abs_value",
    "shap_importance",
    "importance",
]

MODEL_COLUMN_CANDIDATES = ["model", "model_name", "Model"]
FOLD_COLUMN_CANDIDATES = ["fold", "fold_id", "Fold"]


# =========================================================
# HELPERS
# =========================================================

def find_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def identify_model(path):
    path_text = str(path)

    for model in MODEL_NAMES:
        if model.lower() in path_text.lower():
            return model

    return None


def identify_fold(path, df):
    fold_col = find_column(df, FOLD_COLUMN_CANDIDATES)

    if fold_col is not None:
        values = df[fold_col].dropna().unique()

        if len(values) == 1:
            try:
                return int(values[0])
            except (ValueError, TypeError):
                pass

    match = re.search(
        r"fold[_\-\s]?(\d+)",
        str(path),
        flags=re.IGNORECASE,
    )

    if match:
        return int(match.group(1))

    return None


def load_fold_shap_files():
    records = []

    excluded_names = {
        "shap_summary_all_folds.csv",
        "shap_ranking_by_model.csv",
        "stability_summary.csv",
        "stability_pairwise.csv",
    }

    for path in SHAP_DIR.rglob("*.csv"):

        if path.name.lower() in excluded_names:
            continue

        model = identify_model(path)

        if model is None:
            continue

        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"[SKIP] Không đọc được {path.name}: {exc}")
            continue

        feature_col = find_column(df, FEATURE_COLUMNS)
        importance_col = find_column(df, IMPORTANCE_COLUMNS)

        if feature_col is None or importance_col is None:
            continue

        fold = identify_fold(path, df)

        if fold is None:
            continue

        temp = df[[feature_col, importance_col]].copy()
        temp.columns = ["feature", "mean_abs_shap"]

        temp["mean_abs_shap"] = pd.to_numeric(
            temp["mean_abs_shap"],
            errors="coerce",
        )

        temp = temp.dropna(
            subset=["feature", "mean_abs_shap"]
        )

        temp["feature"] = temp["feature"].astype(str)

        temp = (
            temp.groupby("feature", as_index=False)
            ["mean_abs_shap"]
            .mean()
        )

        records.append({
            "model": model,
            "fold": fold,
            "path": path,
            "data": temp,
        })

        print(
            f"[LOAD] {model} | Fold {fold} | "
            f"{len(temp)} features"
        )

    return records


# =========================================================
# STABILITY METRICS
# =========================================================

def calculate_kendall_tau(rank_a, rank_b):
    result = kendalltau(
        rank_a,
        rank_b,
        variant="b",
        nan_policy="omit",
    )

    return float(result.statistic)


def calculate_jaccard(top_a, top_b):
    set_a = set(top_a)
    set_b = set(top_b)

    union = set_a | set_b

    if not union:
        return np.nan

    return len(set_a & set_b) / len(union)


def calculate_pairwise_stability(model, fold_data):
    results = []

    folds = sorted(fold_data.keys())

    for fold_a, fold_b in combinations(folds, 2):

        df_a = fold_data[fold_a]
        df_b = fold_data[fold_b]

        merged = pd.merge(
            df_a,
            df_b,
            on="feature",
            how="outer",
            suffixes=("_a", "_b"),
        ).fillna(0)

        merged["rank_a"] = (
            merged["mean_abs_shap_a"]
            .rank(method="average", ascending=False)
        )

        merged["rank_b"] = (
            merged["mean_abs_shap_b"]
            .rank(method="average", ascending=False)
        )

        tau = calculate_kendall_tau(
            merged["rank_a"],
            merged["rank_b"],
        )

        top_a = (
            df_a.sort_values(
                "mean_abs_shap",
                ascending=False,
            )["feature"]
            .head(TOP_K)
            .tolist()
        )

        top_b = (
            df_b.sort_values(
                "mean_abs_shap",
                ascending=False,
            )["feature"]
            .head(TOP_K)
            .tolist()
        )

        jaccard = calculate_jaccard(top_a, top_b)

        results.append({
            "model": model,
            "fold_a": fold_a,
            "fold_b": fold_b,
            "kendall_tau_b": tau,
            "jaccard_top10": jaccard,
            "top_k": TOP_K,
        })

    return results


# =========================================================
# MAIN
# =========================================================

def main():

    STABILITY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 65)
    print("SHAP EXPLANATION STABILITY")
    print("=" * 65)

    records = load_fold_shap_files()

    if not records:
        raise RuntimeError(
            "Không tìm thấy CSV SHAP theo từng fold. "
            "Kiểm tra tên file, thư mục và các cột feature/mean_abs_shap."
        )

    fold_data = {}

    for record in records:

        model = record["model"]
        fold = record["fold"]

        fold_data.setdefault(model, {})
        fold_data[model][fold] = record["data"]

    pairwise_results = []

    for model in MODEL_NAMES:

        if model not in fold_data:
            print(f"[WARNING] Không có SHAP cho {model}")
            continue

        available_folds = fold_data[model]

        print(
            f"\n[MODEL] {model}: "
            f"{len(available_folds)} folds"
        )

        if len(available_folds) < 2:
            print("[WARNING] Không đủ folds để so sánh.")
            continue

        model_results = calculate_pairwise_stability(
            model,
            available_folds,
        )

        pairwise_results.extend(model_results)

    if not pairwise_results:
        raise RuntimeError(
            "Không tính được stability. "
            "Kiểm tra số fold SHAP đã được tạo."
        )

    pairwise_df = pd.DataFrame(pairwise_results)

    pairwise_path = (
        STABILITY_DIR / "stability_pairwise.csv"
    )

    pairwise_df.to_csv(
        pairwise_path,
        index=False,
    )

    summary_df = (
        pairwise_df.groupby("model")
        .agg(
            n_fold_pairs=("kendall_tau_b", "count"),
            kendall_tau_b_mean=("kendall_tau_b", "mean"),
            kendall_tau_b_std=("kendall_tau_b", "std"),
            jaccard_top10_mean=("jaccard_top10", "mean"),
            jaccard_top10_std=("jaccard_top10", "std"),
        )
        .reset_index()
    )

    summary_path = (
        STABILITY_DIR / "stability_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print("\n" + "=" * 65)
    print("STABILITY SUMMARY")
    print("=" * 65)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(f"\nSaved: {pairwise_path}")
    print(f"Saved: {summary_path}")

    print("\nHoàn thành 03_stability.py")


if __name__ == "__main__":
    main()