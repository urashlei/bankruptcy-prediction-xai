import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

SEED = 42


def load_data():
    X = pd.read_csv("data/processed/X.csv")
    y = pd.read_csv("data/processed/y.csv").iloc[:, 0]
    return X, y


def get_cv():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def make_rf_pipelines():
    def rf(**kw):
        return RandomForestClassifier(random_state=SEED, n_jobs=-1, **kw)

    imp = lambda: SimpleImputer(strategy="median")
    return {
        "Baseline": Pipeline([("imp", imp()), ("clf", rf())]),
        "SMOTE": Pipeline([("imp", imp()), ("smote", SMOTE(random_state=SEED)), ("clf", rf())]),
        "CostSensitive": Pipeline([("imp", imp()), ("clf", rf(class_weight="balanced"))]),
    }