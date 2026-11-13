import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold

from config import (
    DATA_PATH,
    TARGET,
    N_FEATURES,
    N_SPLITS,
    RANDOM_STATE
)


# =====================================================
# LOAD DATASET
# =====================================================

def load_dataset():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dataset: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(
            f"Không tìm thấy cột target: {TARGET}"
        )

    # Target: 0 = Non-bankrupt
    #         1 = Bankrupt

    y = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )

    if y.isna().any():
        raise ValueError(
            "Target chứa giá trị không hợp lệ."
        )

    y = y.astype(int)

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(
            "Target phải có hai nhãn 0 và 1."
        )

    X = df.drop(columns=[TARGET])

    # Convert all financial features to numeric
    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    if X.shape[1] != N_FEATURES:
        raise ValueError(
            f"Expected {N_FEATURES} features, "
            f"found {X.shape[1]}."
        )

    return X, y


# =====================================================
# DATA INFORMATION
# =====================================================

def print_data_information(X, y):

    print("\n========== DATASET INFORMATION ==========")

    print("Number of samples:", len(X))

    print("Number of features:", X.shape[1])

    print("\nClass distribution:")

    print(
        "Non-bankrupt:",
        (y == 0).sum()
    )

    print(
        "Bankrupt:",
        (y == 1).sum()
    )

    print(
        "\nMissing values:",
        X.isna().sum().sum()
    )

    print("=========================================\n")


# =====================================================
# STRATIFIED 5-FOLD
# =====================================================

def create_folds(X, y):

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    return list(cv.split(X, y))


# =====================================================
# FOLD-SPECIFIC IMPUTATION
# =====================================================

def impute_fold(X_train, X_test):

    imputer = SimpleImputer(
        strategy="mean"
    )

    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    return (
        X_train_imputed,
        X_test_imputed,
        imputer
    )