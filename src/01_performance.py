
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "results" / "oof_predictions"
OUTPUT_DIR = PROJECT_DIR / "results" / "performance"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "RF Baseline": "RF_Baseline_predictions.csv",
    "RF SMOTE": "RF_SMOTE_predictions.csv",
    "RF Cost-sensitive": "RF_CostSensitive_predictions.csv",
    "GBT Baseline": "GBT_Baseline_predictions.csv",
    "GBT SMOTE": "GBT_SMOTE_predictions.csv",
    "GBT Cost-sensitive": "GBT_CostSensitive_predictions.csv",
}

RANDOM_SEED = 42
N_BOOTSTRAP = 1000
CI_LEVEL = 0.95

REQUIRED_COLUMNS = [
    "class",
    "prediction(class)",
    "confidence(true)"
]


# ============================================================
# 2. CONVERT RAPIDMINER LABELS
# ============================================================

def convert_binary(series, column_name):

    mapping = {
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0
    }

    values = series.astype(str).str.strip().str.lower()
    result = values.map(mapping)

    if result.isna().any():
        invalid_values = values[result.isna()].unique()

        raise ValueError(
            f"Invalid values in {column_name}: {invalid_values}"
        )

    return result.astype(int).to_numpy()


# ============================================================
# 3. CALCULATE PERFORMANCE METRICS
# ============================================================

def calculate_metrics(y_true, y_pred, y_prob):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    return {
        "Accuracy": accuracy_score(y_true, y_pred),

        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),

        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),

        "F1-score": f1_score(
            y_true, y_pred, zero_division=0
        ),

        "ROC-AUC": roc_auc_score(y_true, y_prob),

        # Average Precision: PR-AUC summary
        "PR-AUC (AP)": average_precision_score(
            y_true, y_prob
        ),

        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }


# ============================================================
# 4. BOOTSTRAP 95% CONFIDENCE INTERVAL
# ============================================================

def bootstrap_confidence_intervals(
    y_true,
    y_pred,
    y_prob,
    n_bootstrap=N_BOOTSTRAP,
    ci_level=CI_LEVEL,
    random_seed=RANDOM_SEED
):

    rng = np.random.default_rng(random_seed)

    n_samples = len(y_true)
    bootstrap_results = []

    for _ in range(n_bootstrap):

        indices = rng.integers(
            0, n_samples, size=n_samples
        )

        y_true_sample = y_true[indices]
        y_pred_sample = y_pred[indices]
        y_prob_sample = y_prob[indices]

        # ROC-AUC and AP require both classes
        if len(np.unique(y_true_sample)) < 2:
            continue

        metrics = calculate_metrics(
            y_true_sample,
            y_pred_sample,
            y_prob_sample
        )

        bootstrap_results.append(metrics)

    if not bootstrap_results:
        raise ValueError(
            "Bootstrap failed: no valid samples."
        )

    bootstrap_df = pd.DataFrame(bootstrap_results)

    alpha = 1 - ci_level

    lower_quantile = alpha / 2
    upper_quantile = 1 - alpha / 2

    ci_results = {}

    metric_names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
        "PR-AUC (AP)"
    ]

    for metric in metric_names:

        ci_results[f"{metric}_CI_Lower"] = (
            bootstrap_df[metric].quantile(lower_quantile)
        )

        ci_results[f"{metric}_CI_Upper"] = (
            bootstrap_df[metric].quantile(upper_quantile)
        )

    ci_results["Valid_Bootstrap_Samples"] = len(
        bootstrap_results
    )

    return ci_results


# ============================================================
# 5. LOAD AND EVALUATE SIX MODELS
# ============================================================

results = []

reference_y_true = None

print("=" * 65)
print("BANKRUPTCY PREDICTION - PERFORMANCE EVALUATION")
print("=" * 65)

for model_name, filename in FILES.items():

    file_path = DATA_DIR / filename

    print(f"\nProcessing: {model_name}")

    if not file_path.exists():
        raise FileNotFoundError(
            f"Cannot find file: {file_path}"
        )

    df = pd.read_csv(file_path, encoding="utf-8-sig")

    # Check required columns
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{model_name} is missing columns: "
            f"{missing_columns}"
        )

    # Convert actual and predicted labels
    y_true = convert_binary(
        df["class"], "class"
    )

    y_pred = convert_binary(
        df["prediction(class)"],
        "prediction(class)"
    )

    # Bankruptcy probability
    y_prob = pd.to_numeric(
        df["confidence(true)"],
        errors="coerce"
    ).to_numpy(dtype=float)

    if not np.isfinite(y_prob).all():
        raise ValueError(
            f"{model_name} has missing or invalid probabilities."
        )

    if not np.logical_and(
        y_prob >= 0,
        y_prob <= 1
    ).all():
        raise ValueError(
            f"{model_name} has probabilities outside [0, 1]."
        )

    if len(y_true) != len(y_pred) or len(y_true) != len(y_prob):
        raise ValueError(
            f"{model_name}: inconsistent array lengths."
        )

    # Check actual labels across all models
    if reference_y_true is None:
        reference_y_true = y_true.copy()

    elif not np.array_equal(reference_y_true, y_true):
        raise ValueError(
            f"{model_name}: actual labels/order differ "
            "from the other model files."
        )

    # Check class distribution
    bankrupt_count = int(np.sum(y_true == 1))
    non_bankrupt_count = int(np.sum(y_true == 0))

    if bankrupt_count == 0 or non_bankrupt_count == 0:
        raise ValueError(
            f"{model_name}: only one target class found."
        )

    # Point estimates
    metrics = calculate_metrics(
        y_true,
        y_pred,
        y_prob
    )

    # Bootstrap confidence intervals
    ci_metrics = bootstrap_confidence_intervals(
        y_true,
        y_pred,
        y_prob
    )

    result = {
        "Model": model_name,
        "N": len(y_true),
        "Bankrupt": bankrupt_count,
        "Non_Bankrupt": non_bankrupt_count,
        **metrics,
        **ci_metrics
    }

    results.append(result)

    print(f"  Samples: {len(y_true)}")
    print(f"  Bankrupt: {bankrupt_count}")
    print(f"  Accuracy: {metrics['Accuracy']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall: {metrics['Recall']:.4f}")
    print(f"  F1-score: {metrics['F1-score']:.4f}")
    print(f"  ROC-AUC: {metrics['ROC-AUC']:.4f}")
    print(f"  PR-AUC (AP): {metrics['PR-AUC (AP)']:.4f}")

    print("  Completed.")


# ============================================================
# 6. SAVE RESULTS
# ============================================================

performance_df = pd.DataFrame(results)

# Save full precision results
output_file = OUTPUT_DIR / "performance_summary.csv"

performance_df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

# Save a compact table for reporting
report_columns = [
    "Model",
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score",
    "ROC-AUC",
    "PR-AUC (AP)",
    "TN",
    "FP",
    "FN",
    "TP"
]

report_df = performance_df[report_columns].copy()

report_file = OUTPUT_DIR / "performance_report.csv"

report_df.to_csv(
    report_file,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 7. DISPLAY SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("PERFORMANCE SUMMARY")
print("=" * 65)

display_columns = [
    "Model",
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score",
    "ROC-AUC",
    "PR-AUC (AP)"
]

print(
    performance_df[display_columns]
    .round(4)
    .to_string(index=False)
)

print("\n" + "=" * 65)
print("CONFIDENCE INTERVALS: 95% BOOTSTRAP")
print("=" * 65)

for _, row in performance_df.iterrows():

    print(f"\n{row['Model']}")

    for metric in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
        "PR-AUC (AP)"
    ]:

        lower = row[f"{metric}_CI_Lower"]
        upper = row[f"{metric}_CI_Upper"]

        print(
            f"  {metric}: "
            f"[{lower:.4f}, {upper:.4f}]"
        )

print("\nSaved files:")
print(f"1. {output_file}")
print(f"2. {report_file}")

print("\nALL SIX MODELS COMPLETED SUCCESSFULLY.")