# company_recommender_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from gensim.models import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

st.set_page_config(page_title="Gợi ý công ty tương tự (Doc2Vec)", layout="wide")

# === CSS, Tên tác giả & Hình ảnh ===
st.markdown(
    """
    <style>
    .css-1d391kg {
        font-size: 16px;
        font-weight: bold;
        position: fixed;
        left: 10px;
    }
    .css-1d391kg-first {
        bottom: 60px;
    }
    .css-1d391kg-second {
        bottom: 30px;
    }
    .image-container {
        position: relative;
        margin-top: 20px;
        text-align: center;
        width: 50%;
        margin-left: auto;
        margin-right: auto;
        padding: 0;
    }
    .image-container img {
        width: 100%;
        height: auto;
        object-fit: cover;
        margin: 0;
        padding: 0;
    }
    .copyright {
        position: fixed;
        top: 10px;
        left: 10px;
        font-size: 14px;
        color: grey;
    }
    .icon {
        position: fixed;
        top: 10px;
        right: 50px;
        font-size: 24px;
        color: grey;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="icon">CopyRight@LeHuuSonHai</div>', unsafe_allow_html=True)

with st.sidebar.container():
    st.markdown('<div class="css-1d391kg-container">', unsafe_allow_html=True)
    st.markdown('<div class="css-1d391kg css-1d391kg-first">Lê Hữu Sơn Hải</div>', unsafe_allow_html=True)
    st.markdown('<div class="css-1d391kg css-1d391kg-second">Đoàn Trung Cường</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.title("🏢 Gợi ý công ty tương tự (Doc2Vec)")
st.image('ITViec.jpg', use_container_width=True)

# === Load data & model ===
@st.cache_resource
def load_data():
    df_companies = pd.read_csv("Data/companies_cleaned.csv")
    doc2vec_model = Doc2Vec.load("models/doc2vec_company.model")
    doc2vec_vectors = np.load("models/doc2vec_vectors.npy")
    xgb_model = joblib.load("models/XGBoost_pipeline.pkl")
    df_overview = pd.read_excel("Data/Overview_Reviews.xlsx")
    df = pd.merge(df_companies, df_overview[['id', 'Recommend working here to a friend']], on='id', how='inner')
    return df, df_companies, doc2vec_model, doc2vec_vectors, xgb_model

df, df_companies, doc2vec_model, doc2vec_vectors, xgboost_classifier = load_data()

stop_words = set(ENGLISH_STOP_WORDS).union({"https", "www"})

def convert_recommend(value):
    if pd.isna(value) or value.strip() == '0%':
        return 0
    try:
        return 1 if int(value.strip('%')) > 50 else 0
    except:
        return np.nan

df['recommend_label'] = df['Recommend working here to a friend'].apply(convert_recommend)

# === Gợi ý công ty ===
def suggest_by_company_name(company_name, top_n=5, industry_filter=None):
    matches = df_companies[df_companies['Company Name'].str.lower().str.contains(company_name.lower())]
    if matches.empty:
        return None, None, None
    idx = matches.index[0]
    vector = doc2vec_vectors[idx]
    sim_scores = cosine_similarity([vector], doc2vec_vectors).flatten()
    sim_scores[idx] = -1
    df_temp = df_companies.copy()
    df_temp['score'] = sim_scores
    if industry_filter:
        df_temp = df_temp[df_temp['Company industry'] == industry_filter]
    top_results = df_temp.sort_values("score", ascending=False).head(top_n)
    return matches.iloc[0]['Company Name'], df_companies.loc[idx]['Company overview'], top_results[['Company Name', 'Company overview', 'score']]

def suggest_by_description(description_text, top_n=5):
    tokens = description_text.lower().split()
    query_vector = doc2vec_model.infer_vector(tokens)
    sim_scores = cosine_similarity([query_vector], doc2vec_vectors)[0]
    top_idx = np.argsort(sim_scores)[::-1][:top_n]
    results = df_companies.loc[top_idx, ['Company Name', 'Company overview']].reset_index(drop=True)
    results['score'] = sim_scores[top_idx]
    return results

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
    return df_temp.sort_values("score", ascending=False).head(top_n)

def predict_recommendation(company_name):
    company_data = df[df['Company Name'].str.lower() == company_name.lower()]
    if company_data.empty:
        return None
    cols = ['Company overview', 'Company industry', 'Training & learning', 'Salary & benefits']
    for col in cols:
        if col not in company_data.columns:
            company_data[col] = ""
        company_data[col] = company_data[col].fillna('')
    try:
        company_data['Salary & benefits'] = pd.to_numeric(company_data['Salary & benefits'], errors='coerce')
    except:
        pass
    try:
        prediction = xgboost_classifier.predict(company_data[cols])
        return "Recommend" if prediction[0] == 1 else "Not Recommend"
    except:
        return "Error"

def plot_recommendation_distribution(df):
    counts = df['recommend_label'].value_counts()
    fig, ax = plt.subplots()
    counts.plot(kind='bar', ax=ax, color=['red', 'green'])
    ax.set_xticklabels(['Not Recommend', 'Recommend'], rotation=0)
    ax.set_title("Tỷ lệ Recommend vs Not Recommend")
    st.pyplot(fig)

def plot_industry_distribution(df):
    plt.figure(figsize=(12, 6))
    ax = sns.countplot(data=df, x='Company industry', hue='recommend_label')
    ax.set_title("Tỷ lệ Recommend theo ngành")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(plt)

def plot_recommendation_ratio_by_industry(df):
    industry_stats = df.groupby("Company industry")["recommend_label"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=industry_stats.index, y=industry_stats.values, ax=ax)
    ax.set_title("Tỷ lệ Recommend theo ngành (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig)

# === Sidebar ===
with st.sidebar:
    top_n = st.slider("Số lượng công ty gợi ý", 3, 15, 5)
    industry_list = df['Company industry'].dropna().unique().tolist()
    selected_industry = st.selectbox("Lọc theo ngành", ["-- Tất cả --"] + sorted(industry_list))
    industry_filter = None if selected_industry == "-- Tất cả --" else selected_industry

# === Tabs ===
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Tên công ty", "✍️ Mô tả", "📊 Phân tích", "📁 Upload"])

with tab1:
    company_input = st.text_input("Nhập tên công ty")
    if company_input:
        found_name, overview, result_df = suggest_by_company_name(company_input, top_n, industry_filter)
        if result_df is not None:
            st.success(f"Tìm thấy: {found_name}")
            st.markdown("**Mô tả:**")
            st.info(overview)
            st.markdown("**Công ty tương tự:**")
            st.dataframe(result_df)
            partners = suggest_partners(company_input, 5)
            if partners is not None:
                st.markdown("**Đối tác khác ngành:**")
                st.dataframe(partners)
            prediction = predict_recommendation(found_name)
            st.markdown(f"**Dự đoán:** {prediction}")

with tab2:
    desc = st.text_area("Nhập mô tả")
    if desc:
        res = suggest_by_description(desc, top_n)
        st.dataframe(res)

with tab3:
    plot_recommendation_distribution(df)
    plot_industry_distribution(df)
    plot_recommendation_ratio_by_industry(df)
 
with tab4:
    uploaded_file = st.file_uploader("Tải file CSV/XLSX", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df_input = pd.read_csv(uploaded_file)
        else:
            df_input = pd.read_excel(uploaded_file)

        st.subheader("📄 Dữ liệu đã tải")
        st.dataframe(df_input)

        required_columns = ['Company overview', 'Company industry', 'Training & learning', 'Salary & benefits']

        # Đảm bảo đủ cột và xử lý giá trị thiếu
        for col in required_columns:
            if col not in df_input.columns:
                df_input[col] = ""

        # Ép kiểu văn bản cho 3 cột text
        text_columns = ['Company overview', 'Company industry', 'Training & learning']
        for col in text_columns:
            df_input[col] = df_input[col].astype(str).fillna("")

        # Ép kiểu số cho cột numeric
        df_input['Salary & benefits'] = pd.to_numeric(df_input['Salary & benefits'], errors='coerce').fillna(0.0)

        # Dự đoán
        try:
            preds = xgboost_classifier.predict(df_input[required_columns])
            df_input['Prediction'] = np.where(preds == 1, "Recommend", "Not Recommend")
        except Exception as e:
            st.error(f"❌ Lỗi khi phân loại: {e}")

        st.subheader("🔍 Kết quả phân loại")
        st.dataframe(df_input)
        st.download_button("⬇️ Tải kết quả", df_input.to_csv(index=False).encode("utf-8"), "batch_predictions.csv")
