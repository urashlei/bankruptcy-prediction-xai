import pandas as pd
import numpy as np
import shap
from pathlib import Path
from scipy.stats import kendalltau
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, recall_score, precision_score, f1_score, average_precision_score
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# 1. Đường dẫn đọc X.csv và y.csv có sẵn trong folder data/processed
PROJECT_ROOT = Path(__file__).resolve().parent.parent
X_PATH = PROJECT_ROOT / "data" / "processed" / "X.csv"
Y_PATH = PROJECT_ROOT / "data" / "processed" / "y.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 2. Đọc dữ liệu
print(f"Đang đọc dữ liệu từ: {X_PATH} ...")
X = pd.read_csv(X_PATH)
y = pd.read_csv(Y_PATH).values.ravel()

SEED = 42

# 3. Tính tỷ lệ trọng số cho Cost-sensitive (Class imbalance)
num_neg = (y == 0).sum()
num_pos = (y == 1).sum()
scale_weight = num_neg / num_pos
print(f"Thống kê Imbalance: Negative (0) = {num_neg}, Positive (1) = {num_pos} | scale_pos_weight = {scale_weight:.2f}")

# 4. Định nghĩa 3 điều kiện XGBoost (Baseline, SMOTE, Cost-sensitive)
pipe_base = ImbPipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('clf', XGBClassifier(random_state=SEED, eval_metric='logloss', n_jobs=-1))
])

pipe_smote = ImbPipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('smote', SMOTE(random_state=SEED)),
    ('clf', XGBClassifier(random_state=SEED, eval_metric='logloss', n_jobs=-1))
])

pipe_cs = ImbPipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('clf', XGBClassifier(scale_pos_weight=scale_weight, random_state=SEED, eval_metric='logloss', n_jobs=-1))
])

pipelines = {
    'XGB-Baseline': pipe_base,
    'XGB-SMOTE': pipe_smote,
    'XGB-CostSensitive': pipe_cs
}

# 5. Đánh giá Performance qua 5-Fold Cross Validation
print("\n--- Đang chạy 5-Fold Cross Validation ---")
scoring = {
    'recall': make_scorer(recall_score, pos_label=1),
    'precision': make_scorer(precision_score, pos_label=1),
    'f1': make_scorer(f1_score, pos_label=1),
    'pr_auc': make_scorer(average_precision_score)
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

perf_list = []
for name, pipe in pipelines.items():
    print(f"Đang chạy cấu hình: {name}...")
    res = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    perf_list.append({
        'Cấu hình': name,
        'Recall': res['test_recall'].mean(),
        'Precision': res['test_precision'].mean(),
        'F1-score': res['test_f1'].mean(),
        'PR-AUC': res['test_pr_auc'].mean()
    })

df_perf = pd.DataFrame(perf_list)

# 6. Tính SHAP Feature Importance và Stability Metrics (Kendall Tau, Jaccard Top-10)
print("\n--- Đang tính SHAP Importance & Độ ổn định (Stability) ---")
imputer = SimpleImputer(strategy='median')
X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

rankings = {}
for name, pipe in pipelines.items():
    pipe.fit(X, y)
    model = pipe.named_steps['clf']

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_imp)

    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    rankings[name] = mean_abs_shap.argsort()[::-1]


def jaccard_top_k(r1, r2, k=10):
    s1, s2 = set(r1[:k]), set(r2[:k])
    return len(s1 & s2) / len(s1 | s2)


base_rank = rankings['XGB-Baseline']
stab_list = []

for cond in ['XGB-SMOTE', 'XGB-CostSensitive']:
    r = rankings[cond]
    tau, _ = kendalltau(base_rank, r)
    jaccard = jaccard_top_k(base_rank, r, k=10)
    stab_list.append({
        'So sánh': f"Baseline vs {cond.replace('XGB-', '')}",
        'Kendall Tau': tau,
        'Jaccard Top-10': jaccard
    })

df_stab = pd.DataFrame(stab_list)

# 7. Xuất kết quả vào folder results/
perf_csv_path = RESULTS_DIR / "xgb_performance_results.csv"
stab_csv_path = RESULTS_DIR / "xgb_stability_results.csv"

df_perf.to_csv(perf_csv_path, index=False)
df_stab.to_csv(stab_csv_path, index=False)

print("\n=================== BẢNG PERFORMANCE ===================")
print(df_perf.to_string(index=False))
print("\n=================== BẢNG STABILITY ====================")
print(df_stab.to_string(index=False))
print(f"\n Hoàn tất! Kết quả đã được xuất vào thư mục: {RESULTS_DIR}")