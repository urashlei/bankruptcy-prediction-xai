import pandas as pd
from scipy.io import arff
import urllib.request
import zipfile
import os

# BƯỚC 1: TẢI VÀ GIẢI NÉN DỮ LIỆU
url = "https://archive.ics.uci.edu/static/public/365/polish+companies+bankruptcy+data.zip"
zip_path = "polish_bankruptcy.zip"
extract_folder = "polish_data"

if not os.path.exists(extract_folder):
    print("Đang tải dữ liệu từ UCI...")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extract('3year.arff', path=extract_folder)
    print("Đã tải và giải nén xong!")


# BƯỚC 2: ĐỌC FILE 3rdYear.arff VÀ CHUYỂN SANG DATAFRAME
file_path = os.path.join(extract_folder, "3year.arff")
print(f"\nĐang đọc file: {file_path}")

data, meta = arff.loadarff(file_path)
df = pd.DataFrame(data)

# Chuyển đổi cột target ('class') từ định dạng byte sang integer (0 và 1)
if df['class'].dtype == 'object':
    df['class'] = df['class'].str.decode('utf-8')
df['class'] = df['class'].astype(int)


# BƯỚC 3: XUẤT BÁO CÁO EDA NHANH
print("\n--- BÁO CÁO EDA SƠ BỘ ---")
print(f"Tổng số quan sát: {df.shape[0]}")
print(f"Tổng số features (không tính target): {df.shape[1] - 1}")

bankrupt_count = df['class'].sum()
normal_count = df.shape[0] - bankrupt_count
print(f"Số mẫu không phá sản (Class 0): {normal_count}")
print(f"Số mẫu phá sản (Class 1): {bankrupt_count}")
print(f"Tỷ lệ mất cân bằng: {bankrupt_count / df.shape[0] * 100:.2f}%")

total_missing = df.isnull().sum().sum()
cols_with_missing = (df.isnull().sum() > 0).sum()
print(f"Tổng số giá trị thiếu (Missing values): {total_missing}")
print(f"Số cột có giá trị thiếu: {cols_with_missing} / {df.shape[1] - 1}")


# BƯỚC 4: LƯU FILE CSV SẠCH (CHƯA ĐIỀN MISSING)
csv_output = "Polish_3rdYear_Cleaned.csv"
df.to_csv(csv_output, index=False)
print(f"\nĐã lưu dữ liệu định dạng CSV tại: {csv_output}")

# Tách thử X và y để kiểm tra
X = df.drop(columns=['class'])
y = df['class']
print("Shape của X:", X.shape)
print("Shape của y:", y.shape)
