import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from pathlib import Path
from gensim.models import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Gợi ý công ty tương tự (Doc2Vec)", layout="wide")

# ================================
# Load mô hình và dữ liệu Doc2Vec
# ================================
@st.cache_resource
def load_doc2vec_and_data():
    df = pd.read_csv("Data/companies_cleaned.csv")
    model = Doc2Vec.load("models/doc2vec_company.model")
    vectors = np.load("models/doc2vec_vectors.npy")
    return df, model, vectors

df_companies, doc2vec_model, doc2vec_vectors = load_doc2vec_and_data()

# ================================
# Load XGBoost Classifier
# ================================
@st.cache_resource
def load_xgboost_classifier():
    with open("saved_models/XGBoost_pipeline.pkl", "rb") as f:
        xgboost_classifier = joblib.load(f)
    return xgboost_classifier

xgboost_classifier = load_xgboost_classifier()

# Stopwords mở rộng
stop_words = set([
    "a", "an", "the", "in", "on", "at", "to", "from", "by", "of", "with",
    "and", "but", "or", "for", "nor", "so", "yet",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "we’re",
    "be", "have", "do", "does", "did", "was", "were", "will", "would", "shall", "should", "may", "might", "can", "could", "must",
    "that", "this", "which", "what", "their", "these", "those", 'https', 'www'
])

# ================================
# Load dữ liệu đánh giá công ty
# ================================
@st.cache_resource
def load_overview_companies():
    df_overview = pd.read_excel("Data/Overview_Reviews.xlsx")
    return df_overview

df_overview = load_overview_companies()
df = pd.merge(df_companies, df_overview[['id', 'Recommend working here to a friend']], on='id', how='inner')

# ================================
# Tạo nhãn phân loại recommend
# ================================
if 'recommend_label' not in df.columns:
    def convert_recommend(value):
        if pd.isna(value) or value.strip() == '0%':
            return 0
        try:
            return 1 if int(value.strip('%')) > 50 else 0
        except:
            return np.nan

    df['recommend_label'] = df['Recommend working here to a friend'].apply(convert_recommend)

# ================================
# Gợi ý theo tên công ty
# ================================
def suggest_by_company_name(company_name, top_n=5, industry_filter=None):
    matches = df_companies[df_companies['Company Name'].str.lower().str.contains(company_name.lower())]
    if matches.empty:
        return None, None, None
    idx = matches.index[0]
    vector = doc2vec_vectors[idx]
    sim_scores = cosine_similarity([vector], doc2vec_vectors).flatten()
    sim_scores[idx] = -1

    df_temp = df_companies.copy()
    df_temp["score"] = sim_scores

    if industry_filter:
        df_temp = df_temp[df_temp["Company industry"] == industry_filter]

    top_results = df_temp.sort_values("score", ascending=False).head(top_n)
    return matches.iloc[0]['Company Name'], df_companies.loc[idx]['Company overview'], top_results[['Company Name', 'Company overview', 'score']]

# ================================
# Gợi ý theo mô tả nội dung
# ================================
def suggest_by_description(description_text, top_n=5):
    tokens = description_text.lower().split()
    query_vector = doc2vec_model.infer_vector(tokens)
    sim_scores = cosine_similarity([query_vector], doc2vec_vectors)[0]
    top_idx = np.argsort(sim_scores)[::-1][:top_n]
    results = df_companies.loc[top_idx, ['Company Name', 'Company overview']].reset_index(drop=True)
    results['score'] = sim_scores[top_idx]
    return results

# ================================
# Gợi ý đối tác khác ngành
# ================================
def suggest_partners(company_name, top_n=5):
    matches = df_companies[df_companies['Company Name'].str.lower().str.contains(company_name.lower())]
    if matches.empty:
        return None
    idx = matches.index[0]
    industry = df_companies.loc[idx, 'Company industry']
    vector = doc2vec_vectors[idx]
    sim_scores = cosine_similarity([vector], doc2vec_vectors).flatten()
    sim_scores[idx] = -1

    df_temp = df_companies.copy()
    df_temp["score"] = sim_scores
    df_temp = df_temp[df_temp["Company industry"] != industry]
    df_temp = df_temp[df_temp["score"] > 0.6]

    top_partners = df_temp.sort_values("score", ascending=False).head(top_n)
    return top_partners[['Company Name', 'Company industry', 'Company overview', 'score']]

# ================================
# Highlight từ tương tự
# ================================
def get_common_keywords(a, b, min_len=4):
    a_words = set(a.lower().split())
    b_words = set(b.lower().split())
    return sorted(list(a_words & b_words - ENGLISH_STOP_WORDS - stop_words), key=lambda x: -len(x))

# ================================
# Phân loại công ty bằng XGBoost
# ================================
def predict_recommendation(company_name):
    company_data = df[df['Company Name'].str.lower() == company_name.lower()]
    if company_data.empty:
        return None

    required_columns = ['Company overview', 'Company industry', 'Training & learning', 'Salary & benefits']
    for col in required_columns:
        if col not in company_data.columns:
            company_data[col] = ""
        company_data.loc[:, col] = company_data[col].fillna('')

    company_data.loc[:, required_columns] = company_data[required_columns].replace("", np.nan)

    try:
        company_data['Salary & benefits'] = pd.to_numeric(company_data['Salary & benefits'], errors='coerce')
    except Exception as e:
        print(f"Error converting 'Salary & benefits' to numeric: {e}")

    features = company_data[required_columns]

    try:
        prediction = xgboost_classifier.predict(features)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error in prediction"

    return "Recommend" if prediction == 1 else "Not Recommend"

# ================================
# Biểu đồ phân tích dữ liệu
# ================================
def plot_recommendation_distribution(df):
    recommendation_counts = df['recommend_label'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    recommendation_counts.plot(kind='bar', ax=ax, color=['green', 'red'])
    ax.set_title("Tỷ lệ Recommend vs Not Recommend")
    ax.set_xlabel("Tình trạng công ty")
    ax.set_ylabel("Số lượng")
    ax.set_xticklabels(['Not Recommend', 'Recommend'], rotation=0)
    st.pyplot(fig)

def plot_industry_distribution(df):
    plt.figure(figsize=(12, 6))
    ax = sns.countplot(data=df, x='Company industry', hue='recommend_label')
    ax.set_title("Tỷ lệ Recommend theo ngành", fontsize=16)
    ax.set_xlabel("Ngành công nghiệp", fontsize=12)
    ax.set_ylabel("Số lượng công ty", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(plt)

# ================================
# Giao diện Streamlit
# ================================
st.title("🏢 Gợi ý công ty tương tự (Doc2Vec)")

st.sidebar.header("🔧 Tuỳ chọn")
top_n = st.sidebar.slider("Số lượng công ty gợi ý", min_value=3, max_value=15, value=5)
industry_list = df['Company industry'].dropna().unique().tolist()
selected_industry = st.sidebar.selectbox("📂 Lọc theo ngành", ["-- Tất cả --"] + sorted(industry_list))
industry_filter = None if selected_industry == "-- Tất cả --" else selected_industry

tab1, tab2 = st.tabs(["🔍 Tìm theo tên công ty", "✍️ Tìm theo mô tả"])

# Tab 1: Tìm theo tên công ty
with tab1:
    st.subheader("🔍 Nhập tên công ty (ví dụ: FPT):")
    company_input = st.text_input("Tên công ty")

    if company_input:
        found_name, overview, result_df = suggest_by_company_name(company_input, top_n=top_n, industry_filter=industry_filter)
        if result_df is not None:
            st.success(f"✅ Tìm thấy: **{found_name}**")
            st.markdown("**📄 Mô tả công ty:**")
            st.info(overview)

            st.markdown("📈 **Các công ty tương tự:**")
            for _, row in result_df.iterrows():
                common = get_common_keywords(overview, row['Company overview'])
                st.markdown(f"🔹 **{row['Company Name']}** – điểm: `{row['score']:.3f}`")
                st.markdown(f"🟩 Từ khóa chung: `{', '.join(common[:10])}`")
                st.markdown("---")

            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Tải danh sách CSV", csv, f"{found_name}_similar_companies.csv", "text/csv")

            st.subheader("🤝 Gợi ý đối tác tiềm năng (khác ngành)")
            partners = suggest_partners(company_input, top_n=5)
            if partners is not None:
                st.dataframe(partners, use_container_width=True)

            recommendation = predict_recommendation(found_name)
            st.subheader("🔎 Kết quả phân loại:")
            st.write(f"Công ty **{found_name}** được phân loại là: **{recommendation}**")

            st.subheader("📊 Biểu đồ phân tích dữ liệu")
            plot_recommendation_distribution(df)
            plot_industry_distribution(df)

# Tab 2: Tìm theo mô tả
with tab2:
    st.subheader("✍️ Nhập mô tả công ty hoặc lĩnh vực bạn muốn tìm:")
    description_input = st.text_area("Ví dụ: Công ty phần mềm chuyên về trí tuệ nhân tạo và dữ liệu lớn...")

    if description_input:
        results_desc = suggest_by_description(description_input, top_n=top_n)
        st.subheader("📋 Danh sách gợi ý theo mô tả:")
        st.dataframe(results_desc, use_container_width=True)

        csv_desc = results_desc.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Tải danh sách CSV", csv_desc, "description_based_suggestions.csv", "text/csv")
