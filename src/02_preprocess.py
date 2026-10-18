import pandas as pd
from pathlib import Path


def preprocess_data():
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    print("Đang đọc dữ liệu thô...")
    try:
        df = pd.read_csv("data/raw/3year.csv")
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy data/raw/3year.csv. Hãy chạy 01_data_check.py trước.")
        return

    target_col = df.columns[-1]          # 'class'
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    print(f"X: {X.shape}, y: {y.shape}")
    print(f"Missing còn lại trong X: {X.isna().sum().sum()} (sẽ impute trong Pipeline, từng fold)")

    # KHÔNG impute ở đây để tránh leakage
    X.to_csv("data/processed/X.csv", index=False)
    y.to_csv("data/processed/y.csv", index=False)
    print("Đã lưu data/processed/X.csv và y.csv")


if __name__ == "__main__":
    preprocess_data()