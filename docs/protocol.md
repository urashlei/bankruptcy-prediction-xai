# QUY TRÌNH THỰC NGHIỆM

## 1. Mục đích

Tài liệu này mô tả quy trình thực hiện nghiên cứu dự đoán phá sản doanh nghiệp bằng các thuật toán học máy, kết hợp phân tích SHAP để đánh giá khả năng giải thích và độ ổn định của thứ hạng đặc trưng.

Quy trình được tổ chức thành hai luồng thực nghiệm:

- **RapidMiner:** Xây dựng, huấn luyện và thực hiện dự đoán Out-of-Fold (OOF) cho sáu cấu hình mô hình. Kết quả OOF được sử dụng để đánh giá hiệu năng dự đoán.
- **Python:** Tái tạo các mô hình theo từng fold để tính SHAP hoặc feature contributions, xếp hạng đặc trưng và đánh giá độ ổn định của lời giải thích.

Hai luồng có mục đích khác nhau. Các mô hình Python được tái tạo nhằm phục vụ phân tích SHAP, không thay thế kết quả huấn luyện và đánh giá hiệu năng chính trên RapidMiner.

## 2. Thiết kế thực nghiệm

### 2.1. Dữ liệu nghiên cứu

Nghiên cứu sử dụng bộ dữ liệu Polish Companies Bankruptcy – 3rd Year.

Các tệp dữ liệu được tổ chức như sau:

| Tệp | Vai trò |
|---|---|
| `data/raw/3year.csv` | Dữ liệu gốc đầu vào. |
| `data/processed/3year_clean.csv` | Dữ liệu sau xử lý, được sử dụng trong các bước thực nghiệm tương ứng. |

Biến mục tiêu là `class`, phân biệt hai nhóm doanh nghiệp:

- `0`: Doanh nghiệp không phá sản.
- `1`: Doanh nghiệp phá sản.

Các đặc trưng đầu vào là các chỉ số tài chính được sử dụng để dự đoán tình trạng phá sản của doanh nghiệp.

### 2.2. Thiết kế chia dữ liệu

Nghiên cứu sử dụng Stratified 5-Fold Cross-Validation để chia dữ liệu thành năm fold.

Các thiết lập chính:

| Tham số | Giá trị |
|---|---|
| Phương pháp chia | Stratified 5-Fold |
| Số fold | 5 |
| Random State | 42 |

Tại mỗi vòng lặp:

- Bốn fold được sử dụng làm tập huấn luyện.
- Một fold được sử dụng làm tập kiểm tra.
- Quá trình được lặp lại cho đến khi cả năm fold đều lần lượt đóng vai trò tập kiểm tra.

Việc phân tầng giúp duy trì tỷ lệ tương đối giữa hai lớp trong các fold.

### 2.3. Sáu cấu hình thực nghiệm

Nghiên cứu so sánh hai họ thuật toán với ba điều kiện xử lý mất cân bằng dữ liệu.

| Họ mô hình | Phương pháp | Định danh trong dự án |
|---|---|---|
| Random Forest | Baseline | `RF_Baseline` |
| Random Forest | SMOTE | `RF_SMOTE` |
| Random Forest | MetaCost | `RF_CostSensitive` |
| Gradient Boosted Trees | Baseline | `GBT_Baseline` |
| Gradient Boosted Trees | SMOTE | `GBT_SMOTE` |
| Gradient Boosted Trees | MetaCost | `GBT_CostSensitive` |

Trong các tên tệp và mã nguồn, `CostSensitive` được giữ nguyên để tương thích với cách đặt tên đã triển khai. Trong nội dung nghiên cứu, phương pháp được gọi là MetaCost.

## 3. Thực nghiệm trên RapidMiner

### 3.1. Các quy trình RapidMiner

Các quy trình thực nghiệm được lưu trong thư mục `rapidminer/`.

| Tệp quy trình | Nội dung |
|---|---|
| `00_CV_Template.rmp` | Quy trình mẫu Cross-Validation. |
| `01_RF_Baseline.rmp` | Random Forest – Baseline. |
| `02_RF_SMOTE.rmp` | Random Forest – SMOTE. |
| `03_RF_CostSensitive.rmp` | Random Forest – MetaCost. |
| `04_GBT_Baseline.rmp` | Gradient Boosted Trees – Baseline. |
| `05_GBT_SMOTE.rmp` | Gradient Boosted Trees – SMOTE. |
| `06_GBT_CostSensitive.rmp` | Gradient Boosted Trees – MetaCost. |

Các quy trình được sử dụng để thực hiện sáu cấu hình thực nghiệm trên cùng thiết kế chia dữ liệu.

### 3.2. Xử lý mất cân bằng dữ liệu

Ba điều kiện được triển khai:

**Baseline**

Sử dụng dữ liệu huấn luyện ban đầu mà không áp dụng phương pháp cân bằng lớp.

**SMOTE**

Áp dụng Synthetic Minority Over-sampling Technique để tạo các mẫu tổng hợp thuộc lớp thiểu số trong tập huấn luyện.

Việc tạo mẫu tổng hợp được thực hiện trên dữ liệu huấn luyện, không thực hiện trên tập kiểm tra.

**MetaCost**

Áp dụng phương pháp học nhạy cảm chi phí nhằm phản ánh mức độ khác nhau giữa chi phí phân loại sai các lớp.

Trong cấu hình nghiên cứu, chi phí dự đoán nhầm doanh nghiệp phá sản thành không phá sản được đặt cao hơn chiều ngược lại, với tỷ lệ chi phí 20:1.

### 3.3. Xuất kết quả dự đoán OOF

Sau khi thực hiện các quy trình trên RapidMiner, kết quả dự đoán Out-of-Fold được xuất thành sáu tệp CSV.

Các tệp được lưu trong:

`results/oof_predictions/`

| Tệp | Cấu hình |
|---|---|
| `RF_Baseline_predictions.csv` | RF – Baseline |
| `RF_SMOTE_predictions.csv` | RF – SMOTE |
| `RF_CostSensitive_predictions.csv` | RF – MetaCost |
| `GBT_Baseline_predictions.csv` | GBT – Baseline |
| `GBT_SMOTE_predictions.csv` | GBT – SMOTE |
| `GBT_CostSensitive_predictions.csv` | GBT – MetaCost |

Mỗi tệp chứa kết quả dự đoán của một cấu hình trên các quan sát được sử dụng làm tập kiểm tra trong Cross-Validation.

Các kết quả này là đầu vào cho bước tính toán hiệu năng bằng Python.

## 4. Đánh giá hiệu năng dự đoán

### 4.1. Tính toán chỉ số hiệu năng

Sau khi có đủ sáu tệp dự đoán OOF, chạy chương trình:

```bash
python src/01_performance.py
```

Chương trình đọc kết quả dự đoán OOF và thực hiện tính toán, tổng hợp các chỉ số đánh giá hiệu năng theo cấu hình mô hình.

Các kết quả được lưu trong:

`results/performance/`

| Tệp | Nội dung |
|---|---|
| `performance_report.csv` | Kết quả đánh giá chi tiết. |
| `performance_summary.csv` | Bảng tổng hợp hiệu năng các cấu hình. |

Khi phân tích kết quả, cần chú ý đến khả năng nhận diện lớp doanh nghiệp phá sản, thay vì chỉ dựa vào độ chính xác tổng thể. Điều này đặc biệt quan trọng do dữ liệu có sự mất cân bằng giữa hai lớp.

## 5. Tái tạo mô hình và phân tích SHAP bằng Python

### 5.1. Mục đích tái tạo mô hình

RapidMiner được sử dụng làm môi trường chính để đánh giá hiệu năng. Tuy nhiên, để thực hiện phân tích SHAP và lưu kết quả theo từng fold, các mô hình được tái tạo bằng Python.

Quá trình tái tạo nhằm tạo ra các mô hình tương ứng với sáu cấu hình và năm fold để phục vụ phân tích khả năng giải thích.

**Lưu ý:** Mô hình Python không được mặc định xem là bản sao hoàn toàn tương đương với mô hình RapidMiner. Sự khác biệt về thư viện, cách triển khai thuật toán và phương pháp xử lý mất cân bằng có thể dẫn đến khác biệt trong kết quả.

### 5.2. Tiền xử lý dữ liệu

Trước khi huấn luyện mô hình Python:

1. Chia dữ liệu theo năm fold tương ứng.
2. Xác định tập huấn luyện và tập kiểm tra của từng fold.
3. Xử lý giá trị thiếu bằng bộ imputer được huấn luyện trên tập huấn luyện.
4. Áp dụng bộ imputer đã huấn luyện lên dữ liệu huấn luyện và dữ liệu kiểm tra tương ứng.
5. Lưu các bộ imputer để sử dụng lại và bảo đảm tính nhất quán trong quá trình xử lý.

Các bộ imputer được lưu trong:

`models/reconstructed/preprocessing/`

Bao gồm:

- `imputer_fold_1.joblib`
- `imputer_fold_2.joblib`
- `imputer_fold_3.joblib`
- `imputer_fold_4.joblib`
- `imputer_fold_5.joblib`

Việc chỉ huấn luyện bộ imputer trên tập huấn luyện giúp hạn chế rò rỉ thông tin từ tập kiểm tra.

### 5.3. Xây dựng mô hình Python

Các hàm xây dựng mô hình và xử lý dữ liệu được tổ chức trong:

- `src/config.py`
- `src/data_utils.py`
- `src/model_builders.py`

Các mô hình được tái tạo cho từng cấu hình và từng fold.

Các mô hình Random Forest được lưu dưới định dạng Joblib. Các mô hình Gradient Boosted Trees sử dụng H2O được lưu theo định dạng mô hình tương ứng của H2O.

Thư mục lưu trữ:

`models/reconstructed/`

Cấu trúc được phân chia theo sáu cấu hình:

- `RF_Baseline/`
- `RF_SMOTE/`
- `RF_CostSensitive/`
- `GBT_Baseline/`
- `GBT_SMOTE/`
- `GBT_CostSensitive/`

### 5.4. Tính giá trị SHAP

Chạy chương trình:

```bash
python src/02_shap_ranking.py
```

Chương trình thực hiện phân tích mức độ đóng góp của các đặc trưng đối với kết quả dự đoán.

Hai cơ chế giải thích được sử dụng:

| Họ mô hình | Phương pháp |
|---|---|
| Random Forest | `shap.TreeExplainer` |
| Gradient Boosted Trees (H2O) | `predict_contributions()` |

Đối với mỗi fold:

1. Tải dữ liệu và mô hình tương ứng.
2. Chuẩn bị dữ liệu giải thích.
3. Tính giá trị SHAP hoặc feature contributions.
4. Tính mức độ quan trọng của từng đặc trưng.
5. Sắp xếp đặc trưng theo mức độ quan trọng giảm dần.
6. Lưu kết quả theo từng cấu hình và fold.

Mức độ quan trọng của đặc trưng được xác định thông qua trung bình giá trị tuyệt đối của đóng góp:

\[
I_j=\frac{1}{n}\sum_{i=1}^{n}|\phi_{ij}|
\]

Trong đó:

- \(I_j\): Mức độ quan trọng của đặc trưng \(j\).
- \(n\): Số quan sát được sử dụng để giải thích.
- \(\phi_{ij}\): Giá trị đóng góp SHAP của đặc trưng \(j\) đối với quan sát \(i\).

Số lượng quan sát tối đa được sử dụng để tính SHAP trong mỗi fold là 1.500.

### 5.5. Lưu kết quả SHAP

Kết quả được lưu trong:

`results/shap/`

Các tệp tổng hợp:

| Tệp | Nội dung |
|---|---|
| `shap_summary_all_folds.csv` | Kết quả SHAP tổng hợp trên các fold. |
| `shap_ranking_by_model.csv` | Thứ hạng đặc trưng theo từng cấu hình mô hình. |

Ngoài ra, mỗi cấu hình có một thư mục riêng chứa kết quả theo từng fold, từ `fold_1_shap.csv` đến `fold_5_shap.csv`.

## 6. Đánh giá độ ổn định thứ hạng đặc trưng

### 6.1. Mục đích

Sau khi thu được thứ hạng đặc trưng SHAP, nghiên cứu đánh giá mức độ nhất quán của thứ hạng trong hai trường hợp:

- Khi thay đổi fold nhưng giữ nguyên cấu hình mô hình.
- Khi thay đổi phương pháp xử lý mất cân bằng dữ liệu nhưng giữ nguyên họ mô hình và fold tương ứng.

Hai hướng đánh giá này phản ánh hai khía cạnh khác nhau của độ ổn định lời giải thích.

### 6.2. Các chỉ số đánh giá

**Kendall's Tau-b**

Đo mức độ tương quan giữa hai bảng xếp hạng đặc trưng.

Giá trị càng gần 1 thể hiện mức độ nhất quán về thứ tự xếp hạng càng cao.

**Jaccard Similarity – Top-10**

Đo mức độ giao nhau giữa hai tập hợp gồm 10 đặc trưng quan trọng nhất:

\[
J(A,B)=\frac{|A\cap B|}{|A\cup B|}
\]

Trong đó:

- \(A\): Tập hợp Top-10 của bảng xếp hạng thứ nhất.
- \(B\): Tập hợp Top-10 của bảng xếp hạng thứ hai.

Giá trị càng cao thể hiện mức độ trùng lặp giữa hai tập Top-10 càng lớn.

### 6.3. Độ ổn định giữa các fold (Cross-fold Stability)

Đối với mỗi cấu hình mô hình:

- So sánh thứ hạng đặc trưng giữa từng cặp fold.
- Với năm fold, có tổng cộng 10 cặp so sánh.
- Tính Kendall's Tau-b và Jaccard Top-10 cho từng cặp.
- Tổng hợp giá trị trung bình và độ lệch chuẩn của các chỉ số.

Sáu cấu hình được đánh giá độc lập.

### 6.4. Độ ổn định giữa các phương pháp (Cross-method Stability)

Đối với mỗi họ thuật toán:

- So sánh Baseline với SMOTE.
- So sánh Baseline với MetaCost.
- So sánh SMOTE với MetaCost.

Việc so sánh được thực hiện trên các fold tương ứng, tức là giữ nguyên fold và thay đổi phương pháp xử lý mất cân bằng dữ liệu.

Với năm fold, mỗi cặp phương pháp có năm lần so sánh.

Kết quả cho biết mức độ thay đổi của thứ hạng đặc trưng khi áp dụng các phương pháp xử lý mất cân bằng khác nhau.

### 6.5. Thực hiện tính toán

Chạy chương trình:

```bash
python src/03_stability.py
```

Các kết quả được lưu trong:

`results/stability/`

| Tệp | Nội dung |
|---|---|
| `stability_pairwise.csv` | Kết quả so sánh chi tiết giữa các cặp fold. |
| `stability_summary.csv` | Bảng tổng hợp Cross-fold Stability. |
| `cross_method_stability_detail.csv` | Kết quả so sánh chi tiết giữa các phương pháp. |
| `cross_method_stability_summary.csv` | Bảng tổng hợp Cross-method Stability. |

## 7. Trực quan hóa kết quả

Sau khi hoàn thành các bước đánh giá hiệu năng, phân tích SHAP và tính toán độ ổn định, chạy:

```bash
python src/04_visualize.py
```

Các biểu đồ được lưu trong:

`results/figures/`

Các hình ảnh kết quả bao gồm:

| Tệp | Nội dung |
|---|---|
| `shap_top10_all_models.png` | So sánh các đặc trưng quan trọng nhất giữa các cấu hình. |
| `kendall_tau_b_comparison.png` | So sánh độ ổn định theo Kendall's Tau-b. |
| `jaccard_top10_comparison.png` | So sánh độ ổn định theo Jaccard Top-10. |
| `cross_method_stability.png` | Trực quan hóa mức độ tương đồng giữa các phương pháp. |
| `combined_stability_comparison.png` | Tổng hợp kết quả so sánh độ ổn định. |

## 8. Thứ tự thực hiện toàn bộ quy trình

Quy trình nghiên cứu được thực hiện theo thứ tự sau:

| Bước | Công cụ / Tệp | Đầu ra |
|---|---|---|
| 1 | Chuẩn bị `3year.csv` | Dữ liệu đầu vào |
| 2 | RapidMiner – sáu quy trình thực nghiệm | Sáu cấu hình mô hình được đánh giá |
| 3 | Xuất kết quả dự đoán OOF | Sáu tệp CSV trong `results/oof_predictions/` |
| 4 | `01_performance.py` | Các bảng đánh giá hiệu năng |
| 5 | `02_shap_ranking.py` | Mô hình tái tạo và kết quả SHAP |
| 6 | `03_stability.py` | Các bảng đánh giá độ ổn định |
| 7 | `04_visualize.py` | Các biểu đồ kết quả |

Tóm tắt luồng thực nghiệm:

```text
DỮ LIỆU ĐẦU VÀO
      │
      ▼
PHÂN CHIA STRATIFIED 5-FOLD
      │
      ├─────────────────────────────┐
      ▼                             ▼
   RAPIDMINER                      PYTHON
      │                             │
      ▼                             ▼
 6 CẤU HÌNH MÔ HÌNH            TIỀN XỬ LÝ DỮ LIỆU
      │                             │
      ▼                             ▼
 DỰ ĐOÁN OOF                 TÁI TẠO 6 CẤU HÌNH
      │                             │
      ▼                             ▼
 TÍNH PERFORMANCE             TÍNH SHAP
      │                             │
      ▼                             ▼
 PERFORMANCE REPORT          SHAP RANKING
                                    │
                                    ▼
                            STABILITY ANALYSIS
                                    │
                                    ▼
                              VISUALIZATION
```

## 9. Các lưu ý về tính tái lập và diễn giải kết quả

### 9.1. Phân biệt kết quả RapidMiner và Python

Hiệu năng dự đoán được tính từ kết quả OOF xuất từ RapidMiner. Trong khi đó, kết quả SHAP được tính từ các mô hình tái tạo bằng Python.

Do khác biệt về môi trường và cách triển khai, không nên mặc định rằng hai hệ thống tạo ra kết quả dự đoán hoặc lời giải thích hoàn toàn giống nhau.

### 9.2. Tránh rò rỉ dữ liệu

Các thao tác tiền xử lý và xử lý mất cân bằng cần được thực hiện trong phạm vi tập huấn luyện của từng fold.

Tập kiểm tra không được sử dụng để:

- Huấn luyện bộ imputer.
- Tạo mẫu tổng hợp bằng SMOTE.
- Huấn luyện mô hình.
- Điều chỉnh tham số dựa trên kết quả kiểm tra.

### 9.3. Diễn giải độ ổn định

Độ ổn định cao không đồng nghĩa với độ chính xác cao.

Một mô hình có thể tạo ra thứ hạng đặc trưng nhất quán giữa các fold nhưng vẫn có hiệu năng dự đoán thấp. Ngược lại, một mô hình có hiệu năng tốt chưa chắc tạo ra lời giải thích ổn định.

Vì vậy, cần xem xét đồng thời kết quả hiệu năng và độ ổn định SHAP.

### 9.4. Giới hạn so sánh giữa hai họ mô hình

Random Forest sử dụng `shap.TreeExplainer`, trong khi Gradient Boosted Trees của H2O sử dụng `predict_contributions()`.

Hai phương pháp có cơ chế tính toán khác nhau. Do đó, cần thận trọng khi so sánh trực tiếp độ lớn giá trị đóng góp SHAP giữa hai họ mô hình.

### 9.5. Giới hạn của đánh giá độ ổn định

Các chỉ số Kendall's Tau-b và Jaccard Top-10 phản ánh mức độ tương đồng giữa các thứ hạng hoặc tập hợp đặc trưng. Chúng không tự chứng minh sự khác biệt có ý nghĩa thống kê.

Do đó, các kết quả cần được diễn giải như bằng chứng định lượng về mức độ nhất quán, không phải kết luận về ý nghĩa thống kê nếu chưa thực hiện kiểm định phù hợp.

## 10. Tài liệu liên quan

Các tài liệu và mã nguồn phục vụ nghiên cứu được tổ chức trong repository:

- `docs/notes.md`: Ghi chú nghiên cứu.
- `docs/protocol.md`: Quy trình thực nghiệm.
- `rapidminer/`: Các quy trình thực nghiệm RapidMiner.
- `src/`: Mã nguồn Python.
- `results/`: Các kết quả thực nghiệm và phân tích.

