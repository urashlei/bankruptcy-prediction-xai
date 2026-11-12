import pandas as pd
from pathlib import Path
from sklearn.model_selection import cross_validate
from common import load_data, get_cv, make_rf_pipelines

SCORING = ["recall", "precision", "f1", "average_precision"]  # average_precision = PR-AUC


def main():
    Path("results/performance").mkdir(parents=True, exist_ok=True)
    X, y = load_data()
    cv = get_cv()

    # Kiểm tra fold: mỗi test fold nên có ~99 mẫu phá sản
    for i, (tr, te) in enumerate(cv.split(X, y), 1):
        print(f"Fold {i}: train={len(tr)} (phá sản={y.iloc[tr].sum()}), "
              f"test={len(te)} (phá sản={y.iloc[te].sum()})")

    rows, per_fold = [], []
    for name, pipe in make_rf_pipelines().items():
        res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING)
        row = {"model": "RandomForest", "condition": name}
        for m in SCORING:
            scores = res[f"test_{m}"]
            row[f"{m}_mean"] = scores.mean()
            row[f"{m}_std"] = scores.std()
            for k, s in enumerate(scores, 1):
                per_fold.append({"model": "RandomForest", "condition": name,
                                 "metric": m, "fold": k, "score": s})
        rows.append(row)
        print(name, {m: round(row[f"{m}_mean"], 3) for m in SCORING})

    pd.DataFrame(rows).to_csv("results/performance/rf_summary.csv", index=False)
    pd.DataFrame(per_fold).to_csv("results/performance/rf_per_fold.csv", index=False)


if __name__ == "__main__":
    main()