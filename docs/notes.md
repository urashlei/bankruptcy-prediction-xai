# Báo cáo EDA & Preprocessing - Dữ liệu 3rdYear


### Thông số dữ liệu gốc
- **Tổng số quan sát:** 10.503 dòng (trong đó có 87 dòng trùng lặp).
- **Phân phối nhãn:** 10.008 mẫu (Class 0) / 495 mẫu (Class 1)
- **Tỷ lệ mất cân bằng:** ~4.71%

### Phân tích giá trị thiếu (Missing Values)
- **Tổng quan:** Tồn tại 9.888 giá trị thiếu (1,47% tổng số ô), xuất hiện tại 44/64 đặc trưng và ảnh hưởng đến 5.618/10.503 dòng (53,5%).
- **Đặc trưng thiếu nghiêm trọng:** Cột `Attr37` thiếu 4.736 giá trị (45,1%).
- **Trạng thái hiện tại:** Đã xử lý (còn 0 missing) trên bản sao phục vụ EDA.

### Nguyên tắc chống rò rỉ dữ liệu (Data Leakage)
- File `3year_clean.csv` CHỈ dùng cho phân tích mô tả.
- Luồng huấn luyện mô hình bắt buộc đọc file thô `3year.csv`. Việc điền khuyết bằng trung vị (median) phải được tính toán và áp dụng cục bộ bên trong tập Training của từng Fold thuộc khối Cross-Validation.
