# 🏢 Company Similarity & Classifier (Doc2Vec + XGBoost)

**Company Similarity and Classifier** là một ứng dụng Streamlit giúp người dùng:

- Gợi ý các công ty tương tự dựa trên tên hoặc mô tả.
- Dự đoán mức độ nên làm việc tại công ty bằng mô hình học máy.
- Trực quan hóa xu hướng ngành và hiệu suất công ty.
- Phân tích hàng loạt từ file đầu vào (CSV/XLSX).

## 🧠 Công nghệ sử dụng

- **Doc2Vec (Gensim):** Tạo vector biểu diễn công ty từ phần mô tả.
- **XGBoost (sklearn):** Phân loại công ty thành *Recommend* / *Not Recommend*.
- **Plotly + Seaborn + Matplotlib:** Trực quan hóa dữ liệu.
- **Streamlit:** Giao diện người dùng trực quan, dễ dùng.

---

## 🎯 Tính năng chính

### 🔍 Tìm công ty tương tự
- Gợi ý top-N công ty tương tự theo tên hoặc mô tả.
- Lọc theo ngành cụ thể.
- Hiển thị từ khóa chung để giải thích tương đồng.

### 🤝 Gợi ý đối tác khác ngành
- Tìm công ty khác ngành có điểm tương đồng cao.

### 📈 Trực quan hóa dữ liệu
- Phân tích phân bố `Recommend` theo ngành.
- Mạng lưới tương tác giữa các công ty.
- Biểu đồ gauge KPI, sunburst, treemap, 3D scatter, animated timeline.

### 📂 Phân loại hàng loạt
- Cho phép tải lên file CSV/XLSX chứa nhiều công ty.
- Trả về dự đoán mức độ *Recommend*.

---

## 📦 Cấu trúc thư mục

```
.
├── Data/
│   ├── companies_cleaned.csv
│   └── Overview_Reviews.xlsx
├── models/
│   ├── doc2vec_company.model
│   ├── doc2vec_vectors.npy
│   └── XGBoost_pipeline.pkl
├── app.py                 # File Streamlit chính
├── README.md              # File mô tả dự án
└── requirements.txt       # Thư viện cần cài
```

---

## ⚙️ Cài đặt & Chạy ứng dụng

### 1. Clone repo:
```bash
git clone https://github.com/yourusername/company-similarity-and-classifier.git
cd company-similarity-and-classifier
```

### 2. Tạo môi trường ảo (khuyến nghị):
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### 3. Cài đặt các thư viện:
```bash
pip install -r requirements.txt
```

### 4. Chạy ứng dụng:
```bash
streamlit run app.py
```

---

## 📄 Yêu cầu file dữ liệu

Bạn cần chuẩn bị các file sau trong thư mục `Data/` và `models/`:

- `companies_cleaned.csv` – Dữ liệu mô tả công ty.
- `Overview_Reviews.xlsx` – Tỷ lệ Recommend và thông tin mở rộng.
- `doc2vec_company.model` – Mô hình Doc2Vec đã huấn luyện.
- `doc2vec_vectors.npy` – Vector biểu diễn công ty.
- `XGBoost_pipeline.pkl` – Pipeline mô hình phân loại.

Liên hệ tác giả nếu bạn cần mô hình mẫu để chạy thử.

---

## 📧 Tác giả

- **Lê Hữu Sơn Hải**  
  Email: lehuusonhai@gmail.com
---

## 📜 License

MIT License – Bạn có thể sử dụng và chỉnh sửa thoải mái cho mục đích cá nhân hoặc nghiên cứu.
