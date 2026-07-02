# Ứng Dụng Huấn Luyện & Dự Báo Machine Learning (Streamlit)

Ứng dụng web được xây dựng tự động từ Jupyter Notebook, cho phép người dùng cấu hình, trực quan hóa, huấn luyện và sử dụng mô hình dự báo trực tiếp trên trình duyệt.

## 📝 Chú thích về mô hình
Đoạn mã được trích xuất từ notebook `5c.csv` dừng lại ở bước chia tách dữ liệu (`train_test_split`), do đó **không có thuật toán cụ thể nào được huấn luyện trong file gốc**. Để ứng dụng có thể chạy hoàn chỉnh chức năng phân loại (`PD` là biến nhị phân), hệ thống đã lựa chọn một thuật toán phân loại phổ biến và đáng tin cậy làm mặc định: **Random Forest Classifier**.

## 🛠️ Yêu cầu & Cài đặt
1. Chắc chắn máy tính của bạn đã cài đặt Python 3.8+.
2. Mở Terminal / Command Prompt và chạy lệnh sau để cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
