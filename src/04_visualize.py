
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from config import ROOT_DIR, SHAP_DIR, STABILITY_DIR, MODEL_NAMES


# =========================================================
# PATHS
# =========================================================

OUTPUT_DIR = ROOT_DIR / "results" / "figures"

RANKING_PATH = SHAP_DIR / "shap_ranking_by_model.csv"
STABILITY_PATH = STABILITY_DIR / "stability_summary.csv"


# =========================================================
# COLUMN HELPERS
# =========================================================

def find_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def load_ranking():

    if not RANKING_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy: {RANKING_PATH}"
        )

    df = pd.read_csv(RANKING_PATH)

    model_col = find_column(
        df,
        ["model", "model_name", "Model"],
    )

    feature_col = find_column(
        df,
        ["feature", "Feature", "attribute", "Attribute"],
    )

    importance_col = find_column(
        df,
        [
            "mean_abs_shap",
            "mean_abs_SHAP",
            "mean_absolute_shap",
            "shap_importance",
            "importance",
        ],
    )

    if not all([model_col, feature_col, importance_col]):
        raise ValueError(
            "Không nhận diện được các cột model, feature, "
            "mean_abs_shap trong shap_ranking_by_model.csv. "
            f"Các cột hiện có: {list(df.columns)}"
        )

    df = df[
        [model_col, feature_col, importance_col]
    ].copy()

    df.columns = [
        "model",
        "feature",
        "mean_abs_shap",
    ]

    df["mean_abs_shap"] = pd.to_numeric(
        df["mean_abs_shap"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["model", "feature", "mean_abs_shap"]
    )

    df["model"] = df["model"].astype(str)
    df["feature"] = df["feature"].astype(str)

    return df


def load_stability():

    if not STABILITY_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy: {STABILITY_PATH}. "
            "Hãy chạy 03_stability.py trước."
        )

    df = pd.read_csv(STABILITY_PATH)

    required_columns = [
        "model",
        "kendall_tau_b_mean",
        "jaccard_top10_mean",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Thiếu cột trong stability_summary.csv: {missing}"
        )

    return df


# =========================================================
# FIGURE 1: SHAP TOP 10
# =========================================================

def plot_shap_top10(ranking_df):

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 11),
    )

    axes = axes.flatten()

    for index, model in enumerate(MODEL_NAMES):

        ax = axes[index]

        model_df = ranking_df[
            ranking_df["model"] == model
        ].copy()

        if model_df.empty:
            ax.set_title(f"{model}\nNo data")
            ax.axis("off")
            continue

        top10 = (
            model_df.sort_values(
                "mean_abs_shap",
                ascending=False,
            )
            .head(10)
            .sort_values(
                "mean_abs_shap",
                ascending=True,
            )
        )

        ax.barh(
            top10["feature"],
            top10["mean_abs_shap"],
        )

        ax.set_title(model)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_ylabel("Feature")

    fig.suptitle(
        "Top 10 SHAP Feature Importance Across Six Models",
        fontsize=15,
    )

    plt.tight_layout()

    output_path = OUTPUT_DIR / "shap_top10_all_models.png"

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"[SAVED] {output_path}")


# =========================================================
# FIGURE 2: KENDALL TAU-B
# =========================================================

def plot_kendall_stability(stability_df):

    df = stability_df.copy()

    df = df.set_index("model").reindex(
        MODEL_NAMES
    ).dropna(
        subset=["kendall_tau_b_mean"]
    ).reset_index()

    fig, ax = plt.subplots(
        figsize=(11, 6),
    )

    ax.bar(
        df["model"],
        df["kendall_tau_b_mean"],
    )

    ax.set_title(
        "Explanation Stability: Kendall's Tau-b"
    )

    ax.set_ylabel("Mean Kendall's Tau-b")
    ax.set_xlabel("Model")
    ax.set_ylim(-1, 1)

    ax.axhline(
        0,
        linewidth=0.8,
    )

    plt.xticks(
        rotation=25,
        ha="right",
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR / "kendall_tau_b_comparison.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"[SAVED] {output_path}")


# =========================================================
# FIGURE 3: JACCARD TOP 10
# =========================================================

def plot_jaccard_stability(stability_df):

    df = stability_df.copy()

    df = df.set_index("model").reindex(
        MODEL_NAMES
    ).dropna(
        subset=["jaccard_top10_mean"]
    ).reset_index()

    fig, ax = plt.subplots(
        figsize=(11, 6),
    )

    ax.bar(
        df["model"],
        df["jaccard_top10_mean"],
    )

    ax.set_title(
        "Explanation Stability: Jaccard Similarity of Top 10"
    )

    ax.set_ylabel("Mean Jaccard Similarity")
    ax.set_xlabel("Model")
    ax.set_ylim(0, 1)

    plt.xticks(
        rotation=25,
        ha="right",
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR / "jaccard_top10_comparison.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"[SAVED] {output_path}")


# =========================================================
# FIGURE 4: COMBINED STABILITY
# =========================================================

def plot_combined_stability(stability_df):

    df = stability_df.copy()

    df = df.set_index("model").reindex(
        MODEL_NAMES
    ).dropna(
        subset=[
            "kendall_tau_b_mean",
            "jaccard_top10_mean",
        ]
    ).reset_index()

    fig, ax = plt.subplots(
        figsize=(12, 6),
    )

    x = range(len(df))
    width = 0.35

    ax.bar(
        [i - width / 2 for i in x],
        df["kendall_tau_b_mean"],
        width,
        label="Kendall Tau-b",
    )

    ax.bar(
        [i + width / 2 for i in x],
        df["jaccard_top10_mean"],
        width,
        label="Jaccard Top-10",
    )

    ax.set_title(
        "Comparison of SHAP Explanation Stability"
    )

    ax.set_ylabel("Stability Score")
    ax.set_xlabel("Model")

    ax.set_xticks(list(x))
    ax.set_xticklabels(
        df["model"],
        rotation=25,
        ha="right",
    )

    ax.set_ylim(0, 1)
    ax.legend()

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR / "combined_stability_comparison.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"[SAVED] {output_path}")


# =========================================================
# MAIN
# =========================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 65)
    print("VISUALIZATION")
    print("=" * 65)

    ranking_df = load_ranking()
    stability_df = load_stability()

    plot_shap_top10(ranking_df)
    plot_kendall_stability(stability_df)
    plot_jaccard_stability(stability_df)
    plot_combined_stability(stability_df)

    print("\nHoàn thành 04_visualize.py")
    print(f"Biểu đồ được lưu tại: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
