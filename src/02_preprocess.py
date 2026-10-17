import pandas as pd
from sklearn.impute import SimpleImputer
from pathlib import Path


def preprocess_data():
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    print("Đang đọc dữ liệu thô...")
    try:
        df = pd.read_csv("data/raw/3year.csv")
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy data/raw/3year.csv. Hãy chạy bước kiểm tra data trước.")
        return


    target_col = df.columns[-1]
    y = df[target_col]
    X = df.drop(columns=[target_col])
    print(f"Đã tách X (Kích thước: {X.shape}) và y (Kích thước: {y.shape})")
    print("Đang xử lý missing values bằng Median...")
    imputer = SimpleImputer(strategy="median")


    X_clean = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X_clean_path = "data/processed/X.csv"
    y_path = "data/processed/y.csv"
    X_clean.to_csv(X_clean_path, index=False)
    y.to_csv(y_path, index=False)

    print(f"Hoàn tất! Đã lưu file sạch tạm thời tại:")
    print(f" - {X_clean_path}")
    print(f" - {y_path}")


if __name__ == "__main__":
    preprocess_data()
