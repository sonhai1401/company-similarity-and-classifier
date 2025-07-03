
# 🏢 Company Similarity & Classifier

This project is an interactive Streamlit dashboard that leverages NLP (Doc2Vec) and machine learning (XGBoost) to suggest similar companies and predict recommendation likelihood based on employee reviews and company profiles.

## 🔍 Features

### 1. Company Name-based Recommendation
- Input a company name to find top-N most similar companies.
- Uses pre-trained **Doc2Vec** embeddings and **cosine similarity**.
- Optional: Filter by industry.
- Display common keywords between companies.

### 2. Description-based Suggestion
- Input free-form text (e.g., “AI-focused software company in fintech”) to find similar companies.
- Uses Doc2Vec `infer_vector` to generate embedding for the input description.

### 3. Partner Suggestion (Cross-industry)
- Finds top potential partner companies from **different industries** but similar profiles.

### 4. Recommendation Classifier (XGBoost)
- Predict whether a company is recommended by employees.
- Uses features like:
  - `Company overview`
  - `Company industry`
  - `Training & learning`
  - `Salary & benefits`

### 5. Batch Prediction via File Upload
- Upload a `.csv` or `.xlsx` file with multiple companies to get predictions in batch.

---

## 📊 Visualizations Dashboard

Includes rich and interactive visualizations:

- 🧭 **Gauge Charts**: Recommend rate, average scores, number of industries
- 🗂️ **Treemap**: Company distribution by industry and recommendation label
- 🌈 **Sunburst Chart**: Industry performance breakdown
- 🕸️ **Interactive Network Graph**: Shows similarity network centered on a selected company
- 🎥 **Animated Timeline**: Simulated growth trends by industry (2020–2024)
- 🎯 **3D Scatter Plot**: Training vs Salary vs Combined Score
- 📊 **Industry-wise Stacked Bars**: Number of companies per recommendation per industry

---

## 🧠 Tech Stack

- **Python 3.9+**
- **Streamlit** for dashboard interface
- **Gensim** for Doc2Vec model
- **XGBoost** for classification
- **Pandas**, **NumPy**, **Plotly**, **NetworkX** for processing and visualization

---

## 📁 Directory Structure

```
.
├── Data/
│   ├── companies_cleaned.csv
│   ├── Overview_Companies.xlsx
│   ├── Overview_Reviews.xlsx
│   └── Reviews.xlsx
├── models/
│   ├── doc2vec_company.model
│   ├── doc2vec_vectors.npy
│   ├── XGBoost_pipeline.pkl
├── app.py                  # Main Streamlit app
├── README.md               # This file
```

## 📂 Dataset Files (`Data/` Folder)

The `Data/` directory contains the necessary datasets used for training and running the app:

| File name               | Description                                           |
|-------------------------|-------------------------------------------------------|
| `companies_cleaned.csv` | Preprocessed company information with textual data.   |
| `Overview_Companies.xlsx` | Raw overview data of companies (extended attributes).|
| `Overview_Reviews.xlsx`  | Contains ratings or review summaries per company.    |
| `Reviews.xlsx`           | Full review texts collected from various sources.    |

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Make sure all files in `Data/` and `models/` are in the correct place.

---

## 👥 Authors

- **Lê Hữu Sơn Hải** – lehuusonhai@gmail.com  
- **Đoàn Trung Cường** – trungcuong.doan2601@gmail.com  

---

## 📜 License

This project is for educational and research purposes only. No commercial use is allowed without permission.

> 🔎 *Ensure all files are placed in the `Data/` directory before running the application.*
