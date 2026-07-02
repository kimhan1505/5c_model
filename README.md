# 应用机器学习: Ứng dụng Dự đoán Kết quả Học tập (PD) Streamlit Web Application

Ứng dụng web này được xây dựng tự động hóa và chuyển đổi quy trình từ cấu trúc Notebook huấn luyện Máy học của bạn sang một giao diện Web tương tác chuyên nghiệp sử dụng thư viện **Streamlit**.

## 📌 Mô tả Ứng dụng & Mô hình Sử dụng
- **Bài toán Machine Learning:** Phân loại nhị phân giám sát (Binary Classification).
- **Thuật toán áp dụng:** `RandomForestClassifier` kế thừa và tối ưu tham số cấu hình chính xác từ kịch bản máy học của notebook gốc.
- **Biến mục tiêu cần dự đoán ($y$):** Cột cột `PD` (Kết quả học tập: giá trị phân loại nhị phân 0 hoặc 1).
- **Tập biến đặc trưng đầu vào ($X$):** Bao gồm 24 biến số khảo sát trích xuất chính xác bao gồm: `TC1` đến `TC5`, `NL1` đến `NL4`, `DK1` đến `DK5`, `V1` đến `V6`, `TS1` đến `TS4`.

## 🛠️ Hướng dẫn cài đặt dự án

### 1. Yêu cầu hệ thống
Đảm bảo máy tính của bạn đã được cài đặt sẵn môi trường Python phiên bản (Khuyến nghị $\ge$ 3.9).

### 2. Cài đặt các thư viện cần thiết
Mở terminal tại thư mục chứa các file và thực thi lệnh sau:
```bash
pip install -r requirements.txt
