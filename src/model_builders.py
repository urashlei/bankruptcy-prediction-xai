
"""
model_builders.py

Reconstruct 6 model configurations used in RapidMiner.
Python models are used ONLY for SHAP and explanation stability.

Models:
1. RF_Baseline
2. RF_SMOTE
3. RF_CostSensitive
4. GBT_Baseline
5. GBT_SMOTE
6. GBT_CostSensitive

IMPORTANT:
- Official predictive performance comes from RapidMiner.
- Python MetaCost is an approximation of the RapidMiner MetaCost operator.
- Python reconstruction is not guaranteed to be identical to RapidMiner.
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE

from config import (
    RF_PARAMS,
    GBT_PARAMS,
    SMOTE_PARAMS,
    COST_MATRIX,
    METACOST_ITERATIONS,
    METACOST_SEED,
    RANDOM_STATE,
)


# ============================================================
# 1. H2O INITIALIZATION
# ============================================================

_H2O_INITIALIZED = False


def get_h2o():
    """
    Initialize H2O once and reuse the same local cluster.
    """

    global _H2O_INITIALIZED

    import h2o

    if not _H2O_INITIALIZED:
        h2o.init(
            nthreads=4,
            max_mem_size="4G",
            port=54321,
        )
        _H2O_INITIALIZED = True

    return h2o


# ============================================================
# 2. SMOTE
# ============================================================

def apply_smote(X_train, y_train):
    """
    Apply SMOTE with normalized feature distances.

    RapidMiner configuration:
    - k = 5
    - normalize = True
    - equalize classes = True
    - append original = True

    StandardScaler is used for neighbor-distance calculation.
    Synthetic data is transformed back to the original scale
    before model training.

    This is a Python approximation of RapidMiner's SMOTE.
    """

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X_train)

    smote = SMOTE(
        k_neighbors=SMOTE_PARAMS["k_neighbors"],
        sampling_strategy=SMOTE_PARAMS["sampling_strategy"],
        random_state=SMOTE_PARAMS["random_state"],
    )

    X_resampled, y_resampled = smote.fit_resample(
        X_scaled,
        np.asarray(y_train),
    )

    X_resampled = scaler.inverse_transform(X_resampled)

    X_resampled = pd.DataFrame(
        X_resampled,
        columns=X_train.columns,
    )

    y_resampled = np.asarray(y_resampled, dtype=int)

    return X_resampled, y_resampled


# ============================================================
# 3. RANDOM FOREST
# ============================================================

def build_random_forest(seed=RANDOM_STATE):
    """
    Build Random Forest using RapidMiner configuration.

    Python equivalent:
    - 100 trees
    - Gini
    - max depth = 10
    - min leaf = 2
    - min split = 4
    - max_features = sqrt
    """

    params = RF_PARAMS.copy()

    params["random_state"] = seed

    return RandomForestClassifier(**params)


# ============================================================
# 4. H2O GRADIENT BOOSTED TREES
# ============================================================

def build_h2o_gbt(seed=RANDOM_STATE):
    """
    Build H2O Gradient Boosting Machine.

    Parameters are mapped from the RapidMiner H2O GBT setup.
    """

    h2o = get_h2o()

    params = GBT_PARAMS.copy()

    params["seed"] = seed

    params["build_tree_one_node"] = True

    return h2o.estimators.H2OGradientBoostingEstimator(
        **params
    )


def fit_h2o_gbt(X_train, y_train, seed=RANDOM_STATE):
    """
    Train H2O GBT on a pandas training dataset.

    Returns:
        model: trained H2O model
    """

    h2o = get_h2o()

    train_df = X_train.copy()

    train_df["class"] = np.asarray(y_train).astype(str)

    train_h2o = h2o.H2OFrame(train_df)

    train_h2o["class"] = train_h2o["class"].asfactor()

    feature_columns = list(X_train.columns)

    model = build_h2o_gbt(seed=seed)

    model.train(
        x=feature_columns,
        y="class",
        training_frame=train_h2o,
    )

    return model


# ============================================================
# 5. METACOST RELABELING
# ============================================================


def calculate_cost_sensitive_labels(
    probabilities,
    cost_matrix=COST_MATRIX,
):
    """
    RapidMiner cost matrix orientation:
        Rows    = predicted class
        Columns = actual class

    Class 0 = non-bankrupt
    Class 1 = bankrupt

    C = [[0, 20],
         [1,  0]]
    """

    probabilities = np.asarray(probabilities)

    p0 = probabilities[:, 0]
    p1 = probabilities[:, 1]

    # Expected cost when predicting class 0
    cost_predict_0 = (
        p0 * cost_matrix[0][0]
        + p1 * cost_matrix[0][1]
    )

    # Expected cost when predicting class 1
    cost_predict_1 = (
        p0 * cost_matrix[1][0]
        + p1 * cost_matrix[1][1]
    )

    # Choose the prediction with minimum expected cost
    new_labels = (
        cost_predict_1 < cost_predict_0
    ).astype(int)

    return new_labels



# ============================================================
# 6. RANDOM FOREST METACOST
# ============================================================

def fit_rf_metacost(X_train, y_train):
    """
    Approximate MetaCost for Random Forest.

    Procedure:
    1. Bootstrap training samples.
    2. Train RF on each bootstrap sample.
    3. Estimate P(class | X) on original training rows.
    4. Average probabilities.
    5. Relabel using the cost matrix.
    6. Train final RF using relabeled data.

    NOTE:
    This is a transparent Python approximation.
    It is not guaranteed to reproduce RapidMiner MetaCost exactly.
    """

    rng = np.random.RandomState(METACOST_SEED)

    y_array = np.asarray(y_train, dtype=int)

    n_samples = len(X_train)

    probability_sum = np.zeros((n_samples, 2))

    for iteration in range(METACOST_ITERATIONS):

        bootstrap_indices = rng.choice(
            n_samples,
            size=n_samples,
            replace=True,
        )

        X_bootstrap = X_train.iloc[bootstrap_indices]

        y_bootstrap = y_array[bootstrap_indices]

        model = build_random_forest(
            seed=METACOST_SEED + iteration
        )

        model.fit(X_bootstrap, y_bootstrap)

        probabilities = model.predict_proba(X_train)

        probability_sum += probabilities

    average_probabilities = (
        probability_sum / METACOST_ITERATIONS
    )

    relabeled_y = calculate_cost_sensitive_labels(
        average_probabilities
    )

    final_model = build_random_forest()

    final_model.fit(X_train, relabeled_y)

    return final_model


# ============================================================
# 7. H2O GBT METACOST
# ============================================================

def predict_h2o_probabilities(model, X):
    """
    Return binary class probabilities from an H2O model.
    """

    h2o = get_h2o()

    frame = h2o.H2OFrame(X)

    predictions = model.predict(frame).as_data_frame()

    if "p1" in predictions.columns:
        return predictions["p1"].to_numpy()

    raise ValueError(
        "Không tìm thấy cột p1 trong kết quả H2O. "
        f"Các cột hiện có: {list(predictions.columns)}"
    )


def fit_gbt_metacost(X_train, y_train):
    """
    Approximate MetaCost for H2O GBT.

    Procedure:
    1. Train bootstrap GBT models.
    2. Estimate probabilities on original training rows.
    3. Average probabilities.
    4. Relabel using the cost matrix.
    5. Train final GBT using relabeled data.

    This requires many GBT training runs and may take time.
    """

    rng = np.random.RandomState(METACOST_SEED)

    y_array = np.asarray(y_train, dtype=int)

    n_samples = len(X_train)

    probability_sum = np.zeros(n_samples)

    for iteration in range(METACOST_ITERATIONS):

        bootstrap_indices = rng.choice(
            n_samples,
            size=n_samples,
            replace=True,
        )

        X_bootstrap = X_train.iloc[bootstrap_indices]

        y_bootstrap = y_array[bootstrap_indices]

        model = fit_h2o_gbt(
            X_bootstrap,
            y_bootstrap,
            seed=METACOST_SEED + iteration,
        )

        probabilities = predict_h2o_probabilities(
            model,
            X_train,
        )

        probability_sum += probabilities

    average_p1 = probability_sum / METACOST_ITERATIONS

    average_probabilities = np.column_stack(
        [1 - average_p1, average_p1]
    )

    relabeled_y = calculate_cost_sensitive_labels(
        average_probabilities
    )

    final_model = fit_h2o_gbt(
        X_train,
        relabeled_y,
        seed=RANDOM_STATE,
    )

    return final_model


# ============================================================
# 8. MAIN MODEL BUILDER
# ============================================================

def fit_model(model_name, X_train, y_train):
    """
    Fit one of the six model configurations.

    Parameters:
        model_name: one of the six model names
        X_train: imputed pandas DataFrame
        y_train: training labels

    Returns:
        Trained sklearn or H2O model.
    """

    valid_models = [
        "RF_Baseline",
        "RF_SMOTE",
        "RF_CostSensitive",
        "GBT_Baseline",
        "GBT_SMOTE",
        "GBT_CostSensitive",
    ]

    if model_name not in valid_models:
        raise ValueError(
            f"Model không hợp lệ: {model_name}"
        )

    y_train = np.asarray(y_train, dtype=int)

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    if model_name == "RF_Baseline":

        model = build_random_forest()

        model.fit(X_train, y_train)

        return model

    if model_name == "RF_SMOTE":

        X_resampled, y_resampled = apply_smote(
            X_train,
            y_train,
        )

        model = build_random_forest()

        model.fit(X_resampled, y_resampled)

        return model

    if model_name == "RF_CostSensitive":

        return fit_rf_metacost(
            X_train,
            y_train,
        )

    # --------------------------------------------------------
    # GRADIENT BOOSTED TREES
    # --------------------------------------------------------

    if model_name == "GBT_Baseline":

        return fit_h2o_gbt(
            X_train,
            y_train,
        )

    if model_name == "GBT_SMOTE":

        X_resampled, y_resampled = apply_smote(
            X_train,
            y_train,
        )

        return fit_h2o_gbt(
            X_resampled,
            y_resampled,
        )

    if model_name == "GBT_CostSensitive":

        return fit_gbt_metacost(
            X_train,
            y_train,
        )
