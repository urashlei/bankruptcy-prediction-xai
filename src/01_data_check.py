import pandas as pd
from scipy.io import arff
import urllib.request
import zipfile
import os
from pathlib import Path


def download_and_check():
    # Tạo thư mục data/raw
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_url = "https://archive.ics.uci.edu/static/public/365/polish+companies+bankruptcy+data.zip"
    zip_path = "polish_bankruptcy.zip"
    arff_file = "3year.arff"
    csv_path = raw_dir / "3year.csv"

    # Tải, giải nén và chuyển đổi nếu file CSV chưa tồn tại
    if not csv_path.exists():
        print("Đang tải dữ liệu từ UCI...")
        urllib.request.urlretrieve(zip_url, zip_path)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            print(f"Đang giải nén {arff_file}...")
            zip_ref.extract(arff_file, path=raw_dir)

        print("Đang chuyển đổi ARFF sang CSV...")
        data, meta = arff.loadarff(raw_dir / arff_file)
        df = pd.DataFrame(data)

        # Decode cột class
        if df['class'].dtype == 'object':
            df['class'] = df['class'].str.decode('utf-8').astype(int)

        df.to_csv(csv_path, index=False)

        # Dọn dẹp file thừa
        os.remove(zip_path)
        os.remove(raw_dir / arff_file)
        print(f"Đã lưu dữ liệu gốc tại: {csv_path}\n")

    # Kiểm tra dữ liệu
    print("--- KIỂM TRA DỮ LIỆU THÔ ---")
    df = pd.read_csv(csv_path)
    print(f"Kích thước dữ liệu: {df.shape}")

    target_col = df.columns[-1]
    bankrupt_count = df[target_col].sum()
    normal_count = df.shape[0] - bankrupt_count

    print(f"Số mẫu không phá sản (Class 0): {normal_count}")
    print(f"Số mẫu phá sản (Class 1): {bankrupt_count}")
    print(f"Tỷ lệ mất cân bằng: {bankrupt_count / df.shape[0] * 100:.2f}%")
    print(f"Tổng số giá trị thiếu (Missing values): {df.isnull().sum().sum()}")


if __name__ == "__main__":
    download_and_check()
