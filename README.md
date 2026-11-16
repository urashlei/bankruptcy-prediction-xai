# DỰ ĐOÁN PHÁ SẢN DOANH NGHIỆP BẰNG MACHINE LEARNING VÀ SHAP

## 1. Giới thiệu dự án

Dự án nghiên cứu ứng dụng các thuật toán học máy (Machine Learning) trong bài toán dự đoán phá sản doanh nghiệp, kết hợp với trí tuệ nhân tạo có khả năng giải thích (Explainable Artificial Intelligence – XAI) nhằm phân tích vai trò của các đặc trưng tài chính đối với kết quả dự đoán.

Nghiên cứu tập trung đánh giá ảnh hưởng của các phương pháp xử lý mất cân bằng dữ liệu đến hai khía cạnh: **hiệu năng dự đoán và độ ổn định của thứ hạng đặc trưng SHAP**.

Hai thuật toán phân loại được sử dụng là Random Forest (RF) và Gradient Boosted Trees (GBT). Mỗi thuật toán được triển khai trong ba điều kiện: Baseline, SMOTE và MetaCost, tạo thành sáu cấu hình thực nghiệm.

Một điểm quan trọng trong quy trình nghiên cứu là RapidMiner được sử dụng làm môi trường chính để huấn luyện và đánh giá hiệu năng mô hình. Python được sử dụng để tính toán các chỉ số hiệu năng từ kết quả dự đoán OOF, tái tạo mô hình phục vụ phân tích SHAP và đánh giá độ ổn định của thứ hạng đặc trưng.

## 2. Mục tiêu nghiên cứu

Dự án hướng đến các mục tiêu sau:

- Đánh giá hiệu năng dự đoán phá sản doanh nghiệp của Random Forest và Gradient Boosted Trees.
- So sánh ảnh hưởng của các phương pháp xử lý mất cân bằng dữ liệu gồm Baseline, SMOTE và MetaCost.
- Xác định các đặc trưng tài chính có mức độ đóng góp lớn vào kết quả dự đoán thông qua SHAP.
- Đánh giá độ ổn định của thứ hạng đặc trưng SHAP giữa các fold.
- So sánh mức độ nhất quán của thứ hạng đặc trưng SHAP giữa các phương pháp xử lý mất cân bằng dữ liệu.

## 3. Dữ liệu nghiên cứu

Nghiên cứu sử dụng bộ dữ liệu **Polish Companies Bankruptcy – 3rd Year**, với tệp dữ liệu đầu vào là `3year.csv`.

Các thông tin cấu hình dữ liệu trong chương trình:

| Thành phần | Giá trị |
|---|---|
| Biến mục tiêu | `class` |
| Lớp không phá sản | 0 |
| Lớp phá sản | 1 |
| Số đặc trưng | 64 |
| Số fold | 5 |
| Random State | 42 |

Dữ liệu được chia thành năm fold theo phương pháp phân tầng (Stratified 5-Fold) nhằm duy trì tỷ lệ giữa hai lớp trong các tập dữ liệu.

## 4. Thiết kế thực nghiệm

Nghiên cứu triển khai sáu cấu hình mô hình:

| Thuật toán | Baseline | SMOTE | MetaCost |
|---|---|---|---|
| Random Forest | RF_Baseline | RF_SMOTE | RF_CostSensitive |
| Gradient Boosted Trees | GBT_Baseline | GBT_SMOTE | GBT_CostSensitive |

Trong mã nguồn, các cấu hình MetaCost vẫn sử dụng định danh `CostSensitive` để tương thích với các tệp và quy trình đã xây dựng.

### 4.1. Random Forest

Random Forest được cấu hình với các tham số chính:

- Số lượng cây: 100.
- Tiêu chí phân chia: Gini.
- Độ sâu tối đa: 10.
- Số mẫu tối thiểu tại nút lá: 2.
- Số mẫu tối thiểu để phân chia nút: 4.
- Số đặc trưng xem xét tại mỗi lần phân chia: Square Root.

### 4.2. Gradient Boosted Trees

Gradient Boosted Trees sử dụng H2O với các tham số chính:

- Số lượng cây: 100.
- Độ sâu tối đa: 5.
- Learning rate: 0.1.
- Số hàng tối thiểu tại nút: 10.
- Số bin: 20.
- Sample rate: 1.0.

### 4.3. Các phương pháp xử lý mất cân bằng dữ liệu

**Baseline:** Sử dụng dữ liệu huấn luyện ban đầu mà không áp dụng phương pháp cân bằng lớp.

**SMOTE:** Sử dụng kỹ thuật Synthetic Minority Over-sampling Technique để tạo thêm các mẫu tổng hợp thuộc lớp thiểu số. Cấu hình trong Python sử dụng `k_neighbors = 5` và `sampling_strategy = 1.0`.

**MetaCost:** Áp dụng phương pháp học nhạy cảm chi phí nhằm xem xét mức độ khác nhau giữa chi phí phân loại sai các lớp. Ma trận chi phí được cấu hình như sau:

| Thực tế / Dự đoán | Không phá sản | Phá sản |
|---|---:|---:|
| Không phá sản | 0 | 20 |
| Phá sản | 1 | 0 |

Trong đó, chi phí dự đoán nhầm doanh nghiệp phá sản thành không phá sản được đặt cao hơn so với chiều ngược lại.

## 5. Công cụ và môi trường thực nghiệm

Dự án sử dụng hai môi trường chính:

**RapidMiner**
- Xây dựng các quy trình tiền xử lý và huấn luyện mô hình.
- Triển khai các cấu hình Random Forest và Gradient Boosted Trees.
- Thực hiện các phương pháp xử lý mất cân bằng dữ liệu.
- Xuất kết quả dự đoán OOF phục vụ đánh giá hiệu năng.

**Python**
- Đọc và xử lý dữ liệu.
- Tính toán các chỉ số hiệu năng từ kết quả dự đoán OOF.
- Tái tạo các mô hình phục vụ phân tích SHAP.
- Tính toán thứ hạng đặc trưng và các chỉ số ổn định.
- Xuất bảng kết quả và dữ liệu phục vụ trực quan hóa.

Các thư viện chính được sử dụng gồm:

- `pandas`, `numpy`: xử lý dữ liệu và tính toán.
- `scikit-learn`: tiền xử lý, chia fold và xây dựng Random Forest.
- `h2o`: xây dựng Gradient Boosted Trees và tính feature contributions.
- `shap`: phân tích khả năng giải thích mô hình.
- `joblib`: lưu và tải mô hình.
- `matplotlib`: trực quan hóa kết quả (nếu được sử dụng trong các script tương ứng).

## 6. Quy trình phân tích SHAP

Phân tích SHAP được thực hiện trên sáu cấu hình mô hình qua năm fold.

Đối với mỗi fold:

1. Chia dữ liệu thành tập huấn luyện và tập kiểm tra theo phân tầng.
2. Thực hiện xử lý giá trị thiếu bằng cách huấn luyện bộ imputer trên tập huấn luyện, sau đó áp dụng cho tập huấn luyện và tập kiểm tra.
3. Huấn luyện lại từng cấu hình mô hình trong Python.
4. Tính giá trị SHAP hoặc feature contributions trên dữ liệu giải thích.
5. Tính mức độ quan trọng của từng đặc trưng bằng trung bình giá trị tuyệt đối của SHAP.
6. Sắp xếp các đặc trưng theo mức độ quan trọng và lưu thứ hạng.

Cụ thể:

- Random Forest sử dụng `shap.TreeExplainer`.
- Gradient Boosted Trees sử dụng phương thức `predict_contributions()` của H2O.
- Số lượng quan sát tối đa được sử dụng để tính SHAP trong mỗi fold là 1.500.
- Cùng một mẫu dữ liệu giải thích được sử dụng cho sáu cấu hình trong cùng một fold.

**Lưu ý về phạm vi:** Các mô hình Python được tái tạo nhằm phục vụ phân tích SHAP, không thay thế các mô hình RapidMiner trong đánh giá hiệu năng chính thức. Do khác biệt về môi trường và cách triển khai, kết quả giải thích SHAP cần được diễn giải trong phạm vi mô hình tái tạo.

## 7. Đánh giá độ ổn định SHAP

Nghiên cứu sử dụng hai chỉ số để đánh giá mức độ nhất quán của thứ hạng đặc trưng.

### 7.1. Kendall's Tau-b

Kendall's Tau-b đo mức độ tương quan giữa hai thứ hạng đặc trưng. Giá trị càng gần 1 thể hiện mức độ nhất quán về thứ tự xếp hạng càng cao.

### 7.2. Jaccard Similarity – Top-10

Jaccard đo mức độ giao nhau giữa hai tập hợp gồm 10 đặc trưng quan trọng nhất:

\[
J(A,B)=\frac{|A\cap B|}{|A\cup B|}
\]

Giá trị càng cao thể hiện hai tập Top-10 có nhiều đặc trưng chung hơn.

### 7.3. Hai hướng đánh giá

**Độ ổn định giữa các fold (Cross-fold Stability)**

- So sánh thứ hạng đặc trưng giữa từng cặp fold trong cùng một cấu hình mô hình.
- Với năm fold, mỗi cấu hình có 10 cặp so sánh.
- Tổng hợp giá trị trung bình và độ lệch chuẩn của Kendall's Tau-b và Jaccard Top-10.

**Độ ổn định giữa các phương pháp (Cross-method Stability)**

- So sánh thứ hạng đặc trưng giữa Baseline, SMOTE và MetaCost trong cùng một họ thuật toán.
- Thực hiện so sánh tại các fold tương ứng.
- Đánh giá mức độ thay đổi thứ hạng đặc trưng khi thay đổi phương pháp xử lý mất cân bằng dữ liệu.

## 8. Cấu trúc thư mục

Cấu trúc thư mục chính của dự án:

```text
bankruptcy-prediction-xai/
│
├── data/
│   ├── raw/
│   │   └── 3year.csv
│   │
│   └── processed/
│       └── 3year_clean.csv
│
├── docs/
│   ├── notes.md
│   └── protocol.md
│
├── models/
│   └── reconstructed/
│       ├── preprocessing/
│       │   ├── imputer_fold_1.joblib
│       │   ├── imputer_fold_2.joblib
│       │   ├── imputer_fold_3.joblib
│       │   ├── imputer_fold_4.joblib
│       │   └── imputer_fold_5.joblib
│       │
│       ├── RF_Baseline/
│       ├── RF_SMOTE/
│       ├── RF_CostSensitive/
│       ├── GBT_Baseline/
│       ├── GBT_SMOTE/
│       └── GBT_CostSensitive/
│
├── rapidminer/
│   ├── 00_CV_Template.rmp
│   ├── 01_RF_Baseline.rmp
│   ├── 02_RF_SMOTE.rmp
│   ├── 03_RF_CostSensitive.rmp
│   ├── 04_GBT_Baseline.rmp
│   ├── 05_GBT_SMOTE.rmp
│   └── 06_GBT_CostSensitive.rmp
│
├── results/
│   ├── figures/
│   │   ├── combined_stability_comparison.png
│   │   ├── cross_method_stability.png
│   │   ├── jaccard_top10_comparison.png
│   │   ├── kendall_tau_b_comparison.png
│   │   └── shap_top10_all_models.png
│   │
│   ├── oof_predictions/
│   │   ├── RF_Baseline_predictions.csv
│   │   ├── RF_SMOTE_predictions.csv
│   │   ├── RF_CostSensitive_predictions.csv
│   │   ├── GBT_Baseline_predictions.csv
│   │   ├── GBT_SMOTE_predictions.csv
│   │   └── GBT_CostSensitive_predictions.csv
│   │
│   ├── performance/
│   │   ├── performance_report.csv
│   │   └── performance_summary.csv
│   │
│   ├── shap/
│   │   ├── shap_ranking_by_model.csv
│   │   ├── shap_summary_all_folds.csv
│   │   ├── RF_Baseline/
│   │   ├── RF_SMOTE/
│   │   ├── RF_CostSensitive/
│   │   ├── GBT_Baseline/
│   │   ├── GBT_SMOTE/
│   │   └── GBT_CostSensitive/
│   │
│   └── stability/
│       ├── stability_pairwise.csv
│       ├── stability_summary.csv
│       ├── cross_method_stability_detail.csv
│       └── cross_method_stability_summary.csv
│
├── src/
│   ├── 01_performance.py
│   ├── 02_shap_ranking.py
│   ├── 03_stability.py
│   ├── 04_visualize.py
│   ├── config.py
│   ├── data_utils.py
│   └── model_builders.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

## 9. Hướng dẫn cài đặt và thực hiện

### 9.1. Clone repository

```bash
git clone <REPOSITORY_URL>
cd bankruptcy-prediction-xai
```

### 9.2. Tạo môi trường ảo

```bash
python -m venv .venv
```

Kích hoạt môi trường trên Windows:

```bash
.venv\Scripts\activate
```

### 9.3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 9.4. Chuẩn bị dữ liệu

Đặt tệp `3year.csv` vào thư mục:

```text
data/raw/3year.csv
```

Đảm bảo tên cột mục tiêu là `class` và dữ liệu có đủ các đặc trưng cần thiết.

### 9.5. Thực hiện đánh giá hiệu năng

Trước hết, thực hiện các quy trình thực nghiệm tương ứng trên RapidMiner và xuất kết quả dự đoán OOF cho sáu cấu hình mô hình.

Sau đó, chạy script tính toán hiệu năng:

```bash
python src/01_performance.py
```

Script sử dụng các tệp dự đoán OOF đã xuất từ RapidMiner để tính toán và tổng hợp các chỉ số đánh giá.

### 9.6. Thực hiện phân tích SHAP

Sau khi chuẩn bị dữ liệu và cấu hình mô hình, chạy:

```bash
python src/02_shap_ranking.py
```

Script tái tạo sáu cấu hình mô hình trên năm fold, tính SHAP hoặc feature contributions và lưu thứ hạng đặc trưng.

### 9.7. Phân tích độ ổn định

Sau khi có kết quả thứ hạng đặc trưng từ các fold, thực hiện các script phân tích độ ổn định tương ứng trong thư mục `src/`.

Các kết quả được lưu trong:

```text
results/stability/
```

Bao gồm dữ liệu chi tiết theo từng cặp so sánh và bảng tổng hợp các chỉ số ổn định.

*Lưu ý: Tên lệnh và thứ tự chạy các script phân tích độ ổn định cần được đối chiếu với các tệp thực tế trong repository.*

## 10. Các tệp kết quả chính

| Tệp | Nội dung |
|---|---|
| `shap_summary_all_folds.csv` | Kết quả SHAP của các đặc trưng trên từng fold và từng cấu hình |
| `shap_ranking_by_model.csv` | Bảng tổng hợp mức độ quan trọng và thứ hạng đặc trưng theo mô hình |
| `stability_pairwise.csv` | Kết quả so sánh độ ổn định giữa từng cặp fold |
| `stability_summary.csv` | Các chỉ số Cross-fold Stability được tổng hợp |
| `cross_method_stability_detail.csv` | Kết quả so sánh giữa các phương pháp xử lý mất cân bằng |
| `cross_method_stability_summary.csv` | Các chỉ số Cross-method Stability được tổng hợp |

Tên tệp được liệt kê theo các kết quả đã được sử dụng trong quá trình phân tích; cần kiểm tra sự tồn tại của từng tệp trong repository trước khi chạy.

## 11. Những hạn chế cần lưu ý

- Hiệu năng dự đoán chính thức được đánh giá trên RapidMiner, trong khi SHAP được tính từ các mô hình tái tạo bằng Python.
- Các mô hình Python không nhất thiết tái tạo hoàn toàn mọi chi tiết triển khai của RapidMiner, đặc biệt đối với SMOTE và MetaCost.
- SHAP của Random Forest và feature contributions của H2O GBT được tính bằng các cơ chế khác nhau; vì vậy cần thận trọng khi so sánh trực tiếp giá trị đóng góp giữa hai họ mô hình.
- Việc sử dụng tối đa 1.500 quan sát trong mỗi fold để tính SHAP có thể ảnh hưởng đến tính đại diện của kết quả giải thích.
- Độ ổn định cao của thứ hạng đặc trưng không đồng nghĩa với việc mô hình dự đoán chính xác hơn hoặc lời giải thích phản ánh quan hệ nhân quả.

## 12. Định hướng nghiên cứu

Dự án hướng đến việc kết hợp đánh giá hiệu năng dự đoán với phân tích khả năng giải thích và độ ổn định của mô hình. Qua đó, nghiên cứu không chỉ xem xét mô hình nào dự đoán tốt hơn mà còn đánh giá mức độ nhất quán của các đặc trưng tài chính quan trọng khi thay đổi phương pháp xử lý mất cân bằng dữ liệu.

---

**Keywords:** Corporate Bankruptcy Prediction; Machine Learning; Random Forest; Gradient Boosted Trees; SMOTE; MetaCost; SHAP; Explainable AI; Feature Ranking Stability.
