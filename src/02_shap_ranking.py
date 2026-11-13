
"""
02_shap_ranking.py

Purpose:
- Reconstruct 6 models on 5 folds.
- Calculate SHAP values for each fold-specific model.
- Save feature importance rankings for stability analysis.

IMPORTANT:
- RapidMiner remains the official performance evaluation environment.
- Python models are reconstructed for SHAP analysis only.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import shap
import joblib

from config import (
    MODEL_DIR,
    SHAP_DIR,
    MODEL_NAMES,
    MAX_EXPLAIN_ROWS,
    RANDOM_STATE,
)

from data_utils import (
    load_dataset,
    create_folds,
    impute_fold,
)

from model_builders import (
    fit_model,
    get_h2o,
)


# ============================================================
# 1. CREATE OUTPUT DIRECTORIES
# ============================================================

SHAP_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. SAMPLE EXPLANATION DATA
# ============================================================

def sample_explanation_data(X_test, fold_number):
    """
    Limit the number of rows used for SHAP calculation.

    The same sampled rows are used across the six models
    within the same fold.
    """

    n_rows = min(MAX_EXPLAIN_ROWS, len(X_test))

    return X_test.sample(
        n=n_rows,
        random_state=RANDOM_STATE + fold_number,
    )


# ============================================================
# 3. EXTRACT POSITIVE-CLASS SHAP VALUES FOR RANDOM FOREST
# ============================================================

def calculate_rf_shap(model, X_explain):
    """
    Calculate SHAP values for the positive class (class = 1).

    Supports different SHAP return formats:
    - List of arrays, one per class.
    - 2D numpy array.
    - 3D numpy array with class dimension.
    """

    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent",
        model_output="raw",
    )

    shap_result = explainer.shap_values(X_explain)

    # Older SHAP versions may return a list of class arrays.
    if isinstance(shap_result, list):

        if len(shap_result) == 2:
            shap_values = np.asarray(shap_result[1])
        else:
            shap_values = np.asarray(shap_result[-1])

    else:

        shap_values = np.asarray(shap_result)

        # Newer SHAP versions may return:
        # (n_samples, n_features, n_classes)
        if shap_values.ndim == 3:

            if shap_values.shape[-1] == 2:
                shap_values = shap_values[:, :, 1]
            else:
                shap_values = shap_values[:, :, -1]

    if shap_values.ndim != 2:
        raise ValueError(
            "SHAP Random Forest có định dạng không hợp lệ: "
            f"{shap_values.shape}"
        )

    return shap_values


# ============================================================
# 4. EXTRACT H2O GBT CONTRIBUTIONS
# ============================================================

def calculate_gbt_shap(model, X_explain):
    """
    Calculate feature contributions using H2O's
    predict_contributions() API.

    H2O contribution values are used for GBT explanation.

    Note:
    H2O contributions are returned on the model's raw
    prediction scale. They should not be interpreted as
    probability-scale SHAP values.
    """

    h2o = get_h2o()

    if not hasattr(model, "predict_contributions"):
        raise RuntimeError(
            "Phiên bản H2O hiện tại không hỗ trợ "
            "predict_contributions() cho model này. "
            "Hãy kiểm tra/cập nhật phiên bản H2O."
        )

    explain_frame = h2o.H2OFrame(X_explain)

    try:

        contribution_frame = model.predict_contributions(
            explain_frame
        )

        contribution_df = contribution_frame.as_data_frame()

    finally:

        h2o.remove(explain_frame)

    # BiasTerm is the model's baseline contribution,
    # not an original financial feature.
    if "BiasTerm" in contribution_df.columns:
        contribution_df = contribution_df.drop(
            columns=["BiasTerm"]
        )

    # Keep exactly the original feature columns and order.
    missing_columns = [
        col for col in X_explain.columns
        if col not in contribution_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Thiếu feature trong kết quả H2O contributions: "
            f"{missing_columns}"
        )

    contribution_df = contribution_df[
        list(X_explain.columns)
    ]

    return contribution_df.to_numpy(dtype=float)


# ============================================================
# 5. CALCULATE FEATURE IMPORTANCE
# ============================================================

def create_feature_importance_table(
    shap_values,
    feature_names,
    model_name,
    fold_number,
):
    """
    Convert SHAP values into a feature importance table.

    Importance = mean absolute SHAP value.
    """

    mean_abs_shap = np.mean(
        np.abs(shap_values),
        axis=0,
    )

    mean_signed_shap = np.mean(
        shap_values,
        axis=0,
    )

    result = pd.DataFrame({
        "model": model_name,
        "fold": fold_number,
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
        "mean_signed_shap": mean_signed_shap,
    })

    result = result.sort_values(
        by="mean_abs_shap",
        ascending=False,
    ).reset_index(drop=True)

    result["rank"] = np.arange(
        1,
        len(result) + 1,
    )

    return result


# ============================================================
# 6. SAVE FOLD MODEL
# ============================================================

def save_fold_model(model, model_name, fold_number):
    """
    Save reconstructed model for future inspection.

    sklearn models -> joblib
    H2O models -> h2o.save_model()
    """

    output_dir = MODEL_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_name.startswith("RF_"):

        output_path = (
            output_dir / f"fold_{fold_number}.joblib"
        )

        joblib.dump(model, output_path)

    else:

        h2o = get_h2o()

        h2o.save_model(
            model=model,
            path=str(output_dir),
            force=True,
        )


# ============================================================
# 7. PROCESS ONE MODEL ON ONE FOLD
# ============================================================

def process_one_model(
    model_name,
    fold_number,
    X_train,
    y_train,
    X_test,
):
    """
    Train one model and calculate SHAP for one fold.
    """

    print(
        f"\n[{model_name}] "
        f"Fold {fold_number}: training..."
    )

    model = fit_model(
        model_name,
        X_train,
        y_train,
    )

    # Save reconstructed model.
    save_fold_model(
        model,
        model_name,
        fold_number,
    )

    # Sample the test fold for SHAP calculation.
    X_explain = sample_explanation_data(
        X_test,
        fold_number,
    )

    print(
        f"[{model_name}] Fold {fold_number}: "
        f"calculating SHAP on {len(X_explain)} rows..."
    )

    if model_name.startswith("RF_"):

        shap_values = calculate_rf_shap(
            model,
            X_explain,
        )

    else:

        shap_values = calculate_gbt_shap(
            model,
            X_explain,
        )

    importance_df = create_feature_importance_table(
        shap_values=shap_values,
        feature_names=list(X_explain.columns),
        model_name=model_name,
        fold_number=fold_number,
    )

    # Save feature ranking for this fold.
    output_dir = SHAP_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_dir / f"fold_{fold_number}_shap.csv"
    )

    importance_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"[{model_name}] Fold {fold_number}: "
        f"saved {output_path}"
    )

    # Release H2O model after its contribution calculation.
    if model_name.startswith("GBT_"):
        get_h2o().remove(model)

    return importance_df


# ============================================================
# 8. RUN ALL SIX MODELS ON FIVE FOLDS
# ============================================================

def main():

    print("=" * 65)
    print("SHAP ANALYSIS - BANKRUPTCY PREDICTION")
    print("=" * 65)

    # Load original dataset.
    X, y = load_dataset()

    print(f"Number of observations: {len(X)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Bankrupt companies: {(y == 1).sum()}")
    print(f"Non-bankrupt companies: {(y == 0).sum()}")

    # Create five stratified folds.
    folds = create_folds(X, y)

    all_results = []

    # Process each fold.
    for fold_index, (train_idx, test_idx) in enumerate(
        folds,
        start=1,
    ):

        print("\n" + "=" * 65)
        print(f"PROCESSING FOLD {fold_index}/5")
        print("=" * 65)

        X_train_raw = X.iloc[train_idx]
        X_test_raw = X.iloc[test_idx]

        y_train = y.iloc[train_idx]

        # Fit imputer on training fold only.
        X_train, X_test, imputer = impute_fold(
            X_train_raw,
            X_test_raw,
        )

        # Save imputer for reproducibility.
        imputer_dir = MODEL_DIR / "preprocessing"
        imputer_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            imputer,
            imputer_dir / f"imputer_fold_{fold_index}.joblib",
        )

        # Process all six model configurations.
        for model_name in MODEL_NAMES:

            importance_df = process_one_model(
                model_name=model_name,
                fold_number=fold_index,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
            )

            all_results.append(importance_df)

    # --------------------------------------------------------
    # COMBINE ALL FOLD RESULTS
    # --------------------------------------------------------

    if not all_results:
        raise RuntimeError("Không có kết quả SHAP nào được tạo.")

    all_results_df = pd.concat(
        all_results,
        ignore_index=True,
    )

    all_results_path = (
        SHAP_DIR / "shap_summary_all_folds.csv"
    )

    all_results_df.to_csv(
        all_results_path,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # AGGREGATE MEAN ABSOLUTE SHAP BY MODEL AND FEATURE
    # --------------------------------------------------------

    ranking_df = (
        all_results_df
        .groupby(
            ["model", "feature"],
            as_index=False,
        )
        .agg(
            mean_abs_shap=("mean_abs_shap", "mean"),
            std_abs_shap=("mean_abs_shap", "std"),
            mean_signed_shap=("mean_signed_shap", "mean"),
        )
    )

    ranking_df["rank"] = (
        ranking_df
        .groupby("model")["mean_abs_shap"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    ranking_df = ranking_df.sort_values(
        ["model", "rank"]
    ).reset_index(drop=True)

    ranking_path = (
        SHAP_DIR / "shap_ranking_by_model.csv"
    )

    ranking_df.to_csv(
        ranking_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 65)
    print("SHAP ANALYSIS COMPLETED")
    print("=" * 65)

    print(f"All fold results: {all_results_path}")
    print(f"Aggregated ranking: {ranking_path}")

    print("\nTop 10 features by model:")

    for model_name in MODEL_NAMES:

        top_10 = (
            ranking_df[
                ranking_df["model"] == model_name
            ]
            .head(10)
        )

        print(f"\n{model_name}")
        print(
            top_10[
                ["rank", "feature", "mean_abs_shap"]
            ].to_string(index=False)
        )

    # Shut down H2O after all models are processed.
    get_h2o().shutdown(prompt=False)


if __name__ == "__main__":
    main()
