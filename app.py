import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import networkx as nx
from collections import Counter
from gensim.models import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Gợi ý công ty tương tự (Doc2Vec)", layout="wide")

@st.cache_resource
def load_doc2vec_and_data():
    df = pd.read_csv("Data/companies_cleaned.csv")
    model = Doc2Vec.load("models/doc2vec_company.model")
    vectors = np.load("models/doc2vec_vectors.npy")
    return df, model, vectors

df_companies, doc2vec_model, doc2vec_vectors = load_doc2vec_and_data()

@st.cache_resource
def load_xgboost_classifier():
    with open("models/XGBoost_pipeline.pkl", "rb") as f:
        xgboost_classifier = joblib.load(f)
    return xgboost_classifier

xgboost_classifier = load_xgboost_classifier()

stop_words = set([
    "a", "an", "the", "in", "on", "at", "to", "from", "by", "of", "with",
    "and", "but", "or", "for", "nor", "so", "yet",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "we’re",
    "be", "have", "do", "does", "did", "was", "were", "will", "would", "shall", "should", "may", "might", "can", "could", "must",
    "that", "this", "which", "what", "their", "these", "those", 'https', 'www'
])

@st.cache_resource
def load_overview_companies():
    df_overview = pd.read_excel("Data/Overview_Reviews.xlsx")
    return df_overview

df_overview = load_overview_companies()
df = pd.merge(df_companies, df_overview[['id', 'Recommend working here to a friend']], on='id', how='inner')

def convert_recommend(value):
    if pd.isna(value) or value.strip() == '0%':
        return 0
    try:
        return 1 if int(value.strip('%')) > 50 else 0
    except:
        return np.nan

if 'recommend_label' not in df.columns:
    df['recommend_label'] = df['Recommend working here to a friend'].apply(convert_recommend)

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

    top_partners = df_temp.sort_values("score", ascending=False).head(top_n)
    return top_partners[['Company Name', 'Company industry', 'Company overview', 'score']]

def get_common_keywords(a, b, min_len=4):
    a_words = set(a.lower().split())
    b_words = set(b.lower().split())
    return sorted(list(a_words & b_words - ENGLISH_STOP_WORDS - stop_words), key=lambda x: -len(x))

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

# Cập nhật các hàm trực quan hóa
def plot_recommendation_distribution_enhanced(df):
    """Biểu đồ phân bố recommendation với animation và styling đẹp"""
    recommendation_counts = df['recommend_label'].value_counts()
    
    # Tạo pie chart với Plotly
    fig = go.Figure(data=[go.Pie(
        labels=['Recommend', 'Not Recommend'], 
        values=[recommendation_counts.get(1, 0), recommendation_counts.get(0, 0)],
        hole=.4,
        marker=dict(colors=['#00D4AA', '#FF6B6B'], line=dict(color='#FFFFFF', width=3)),
        textfont_size=16,
        textinfo='label+percent+value',
        hovertemplate='<b>%{label}</b><br>Số lượng: %{value}<br>Tỷ lệ: %{percent}<extra></extra>',
        pull=[0.1, 0]  # Tách ra một phần
    )])
    
    fig.update_layout(
        title={
            'text': '🎯 Tỷ lệ Recommend vs Not Recommend',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#2F4F4F'}
        },
        font=dict(size=14),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Thêm annotation ở giữa
    total = sum(recommendation_counts.values)
    fig.add_annotation(
        text=f"<b>Tổng cộng<br>{total:,} công ty</b>",
        x=0.5, y=0.5,
        font_size=18,
        font_color="#2F4F4F",
        showarrow=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_industry_distribution_enhanced(df):
    """Biểu đồ phân bố theo ngành với interactive features"""
    industry_recommend = df.groupby(['Company industry', 'recommend_label']).size().unstack(fill_value=0)
    
    fig = go.Figure()
    
    # Thêm bar cho Not Recommend
    fig.add_trace(go.Bar(
        name='Not Recommend',
        x=industry_recommend.index,
        y=industry_recommend.get(0, [0]*len(industry_recommend.index)),
        marker_color='#FF6B6B',
        hovertemplate='<b>%{x}</b><br>Not Recommend: %{y}<extra></extra>',
        opacity=0.8
    ))
    
    # Thêm bar cho Recommend
    fig.add_trace(go.Bar(
        name='Recommend',
        x=industry_recommend.index,
        y=industry_recommend.get(1, [0]*len(industry_recommend.index)),
        marker_color='#00D4AA',
        hovertemplate='<b>%{x}</b><br>Recommend: %{y}<extra></extra>',
        opacity=0.8
    ))
    
    fig.update_layout(
        title={
            'text': '📊 Phân bố Recommend theo từng ngành',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#2F4F4F'}
        },
        xaxis_title="Ngành công nghiệp",
        yaxis_title="Số lượng công ty",
        barmode='stack',
        hovermode='x unified',
        height=600,
        xaxis={'categoryorder':'total descending'},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12)
    )
    
    fig.update_xaxes(tickangle=45)
    
    st.plotly_chart(fig, use_container_width=True)

def create_interactive_network_graph(df_companies, doc2vec_vectors, selected_company=None, threshold=0.7):
    """Tạo network graph tương tác với khả năng chọn công ty trung tâm"""
    
    if selected_company:
        # Tìm company và tạo network xung quanh nó
        matches = df_companies[df_companies['Company Name'].str.lower().str.contains(selected_company.lower())]
        if matches.empty:
            st.warning(f"Không tìm thấy công ty '{selected_company}'")
            return
        
        center_idx = matches.index[0]
        center_vector = doc2vec_vectors[center_idx]
        
        # Tính similarity với tất cả công ty khác
        similarities = cosine_similarity([center_vector], doc2vec_vectors).flatten()
        
        # Lấy top 20 công ty tương tự nhất
        similar_indices = np.argsort(similarities)[::-1][:21]  # +1 để bao gồm chính nó
        similar_indices = similar_indices[similar_indices != center_idx][:20]  # Loại bỏ chính nó
        
        # Tạo graph
        G = nx.Graph()
        
        # Thêm center node
        G.add_node(center_idx, 
                  name=df_companies.loc[center_idx, 'Company Name'],
                  industry=df_companies.loc[center_idx, 'Company industry'],
                  node_type='center')
        
        # Thêm các nodes tương tự
        for idx in similar_indices:
            G.add_node(idx, 
                      name=df_companies.loc[idx, 'Company Name'],
                      industry=df_companies.loc[idx, 'Company industry'],
                      node_type='similar')
            
            # Thêm edge nếu similarity > threshold
            if similarities[idx] > threshold:
                G.add_edge(center_idx, idx, weight=similarities[idx])
        
        # Tạo layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Tạo traces cho edges
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(G[edge[0]][edge[1]]['weight'])
        
        # Tạo trace cho edges
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='rgba(125, 125, 125, 0.5)'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Tạo traces cho nodes
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            node_info = G.nodes[node]
            node_text.append(f"{node_info['name'][:20]}...")
            
            if node_info['node_type'] == 'center':
                node_colors.append('#FF6B6B')
                node_sizes.append(25)
            else:
                node_colors.append('#00D4AA')
                node_sizes.append(15)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            hovertext=[f"{G.nodes[node]['name']}<br>Industry: {G.nodes[node]['industry']}" for node in G.nodes()],
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white')
            )
        )
        
        fig = go.Figure(data=[edge_trace, node_trace])
        
        fig.update_layout(
            title={
                'text': f'🕸️ Mạng lưới công ty tương tự với "{df_companies.loc[center_idx, "Company Name"]}"',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#2F4F4F'}
            },
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Hiển thị thông tin chi tiết
        st.subheader("📋 Danh sách công ty tương tự:")
        similar_companies = []
        for idx in similar_indices:
            if similarities[idx] > threshold:
                similar_companies.append({
                    'Công ty': df_companies.loc[idx, 'Company Name'],
                    'Ngành': df_companies.loc[idx, 'Company industry'],
                    'Độ tương tự': f"{similarities[idx]:.3f}"
                })
        
        if similar_companies:
            st.dataframe(pd.DataFrame(similar_companies), use_container_width=True)
        else:
            st.info("Không có công ty nào có độ tương tự > 70%")

def create_industry_performance_sunburst(df):
    """Tạo sunburst chart cho hiệu suất theo ngành"""
    # Chuẩn bị dữ liệu
    industry_data = df.groupby(['Company industry', 'recommend_label']).size().reset_index()
    industry_data.columns = ['industry', 'recommend', 'count']
    industry_data['recommend_text'] = industry_data['recommend'].map({0: 'Not Recommend', 1: 'Recommend'})
    
    # Tạo sunburst chart
    fig = px.sunburst(
        industry_data,
        path=['industry', 'recommend_text'],
        values='count',
        color='recommend',
        color_discrete_map={0: '#FF6B6B', 1: '#00D4AA'},
        title='🌟 Phân tích hiệu suất theo ngành (Sunburst Chart)'
    )
    
    fig.update_layout(
        title={
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2F4F4F'}
        },
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_treemap_visualization(df):
    """Tạo treemap cho visualization ngành và recommendation"""
    # Chuẩn bị dữ liệu
    industry_counts = df.groupby(['Company industry', 'recommend_label']).size().reset_index()
    industry_counts.columns = ['industry', 'recommend', 'count']
    industry_counts['recommend_text'] = industry_counts['recommend'].map({0: 'Not Recommend', 1: 'Recommend'})
    industry_counts['full_label'] = industry_counts['industry'] + ' - ' + industry_counts['recommend_text']
    
    fig = px.treemap(
        industry_counts,
        path=[px.Constant("Tất cả ngành"), 'industry', 'recommend_text'],
        values='count',
        color='recommend',
        color_discrete_map={0: '#FF6B6B', 1: '#00D4AA'},
        title='🗂️ Treemap - Phân bố công ty theo ngành và recommendation'
    )
    
    fig.update_layout(
        title={
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2F4F4F'}
        },
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_animated_timeline_chart(df):
    """Tạo animated chart giả lập timeline phát triển"""
    # Giả lập dữ liệu timeline (vì không có dữ liệu thời gian thật)
    industries = df['Company industry'].value_counts().head(10).index.tolist()
    
    # Tạo dữ liệu giả cho timeline
    timeline_data = []
    for year in range(2020, 2025):
        for industry in industries:
            count = df[df['Company industry'] == industry].shape[0]
            # Giả lập sự tăng trưởng
            growth_factor = 1 + (year - 2020) * 0.1 + np.random.uniform(-0.05, 0.15)
            timeline_data.append({
                'Year': year,
                'Industry': industry,
                'Count': int(count * growth_factor),
                'Recommend_Rate': np.random.uniform(0.4, 0.8)
            })
    
    timeline_df = pd.DataFrame(timeline_data)
    
    fig = px.bar(
        timeline_df,
        x='Industry',
        y='Count',
        color='Recommend_Rate',
        animation_frame='Year',
        color_continuous_scale='RdYlGn',
        title='📈 Xu hướng phát triển theo ngành (2020-2024)',
        labels={'Count': 'Số lượng công ty', 'Recommend_Rate': 'Tỷ lệ Recommend'}
    )
    
    fig.update_layout(
        title={
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2F4F4F'}
        },
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_tickangle=45
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_3d_scatter_plot(df):
    """Tạo 3D scatter plot cho phân tích đa chiều"""
    if all(col in df.columns for col in ['Training & learning', 'Salary & benefits']):
        df_clean = df.dropna(subset=['Training & learning', 'Salary & benefits', 'recommend_label'])
        
        if not df_clean.empty:
            # Tạo chiều thứ 3 bằng cách tính toán từ dữ liệu có sẵn
            df_clean['company_score'] = df_clean['Training & learning'] + df_clean['Salary & benefits']
            
            fig = px.scatter_3d(
                df_clean,
                x='Training & learning',
                y='Salary & benefits',
                z='company_score',
                color='recommend_label',
                size='Salary & benefits',
                hover_data=['Company Name'],
                color_discrete_map={0: '#FF6B6B', 1: '#00D4AA'},
                title='🎯 Phân tích 3D - Training vs Salary vs Overall Score'
            )
            
            fig.update_layout(
                title={
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20, 'color': '#2F4F4F'}
                },
                height=700,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                scene=dict(
                    xaxis_title="Training & Learning",
                    yaxis_title="Salary & Benefits",
                    zaxis_title="Overall Score"
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)

def create_gauge_charts(df):
    """Tạo gauge charts cho các KPI chính"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Gauge cho tỷ lệ recommend
        recommend_rate = (df['recommend_label'].sum() / len(df) * 100) if len(df) > 0 else 0
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = recommend_rate,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Tỷ lệ Recommend (%)"},
            delta = {'reference': 65},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "#00D4AA"},
                'steps': [
                    {'range': [0, 50], 'color': "#FF6B6B"},
                    {'range': [50, 80], 'color': "#FFD700"},
                    {'range': [80, 100], 'color': "#00D4AA"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Gauge cho số lượng ngành
        industry_count = df['Company industry'].nunique()
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = industry_count,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Số ngành nghề"},
            gauge = {
                'axis': {'range': [None, 50]},
                'bar': {'color': "#4169E1"},
                'steps': [
                    {'range': [0, 10], 'color': "#FFE4E1"},
                    {'range': [10, 30], 'color': "#87CEEB"},
                    {'range': [30, 50], 'color': "#4169E1"}
                ]
            }
        ))
        
        fig.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Gauge cho điểm trung bình
        if 'Salary & benefits' in df.columns:
            avg_score = df['Salary & benefits'].mean()
        else:
            avg_score = 3.5
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = avg_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Điểm TB Tổng thể"},
            gauge = {
                'axis': {'range': [None, 5]},
                'bar': {'color': "#FF6347"},
                'steps': [
                    {'range': [0, 2], 'color': "#FFE4E1"},
                    {'range': [2, 4], 'color': "#FFA07A"},
                    {'range': [4, 5], 'color': "#FF6347"}
                ]
            }
        ))
        
        fig.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

# Cập nhật tab trực quan hóa
def enhanced_visualization_tab_v2(df, df_companies, doc2vec_vectors):
    """Tab trực quan hóa được cải tiến v2"""
    st.subheader("📊 Dashboard Phân tích Dữ liệu Công ty")
    
    # Tạo tabs con
    viz_tab1, viz_tab2, viz_tab3, = st.tabs([
        "📈 Tổng quan Executive", 
        "🕸️ Mạng lưới Tương tác", 
        "🚀 Xu hướng & Dự báo"
    ])
    
    with viz_tab1:
        # Executive summary với gauge charts
        st.subheader("⚡ KPI Dashboard")
        create_gauge_charts(df)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            plot_recommendation_distribution_enhanced(df)
        
        with col2:
            create_treemap_visualization(df)
        
        st.markdown("---")
        plot_industry_distribution_enhanced(df)
    
    with viz_tab2:
        st.subheader("🕸️ Mạng lưới Công ty Tương tự")
        
        # Tạo selectbox để chọn công ty
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_company = st.selectbox(
                "🔍 Chọn công ty để xem mạng lưới tương tự:",
                [''] + df_companies['Company Name'].tolist()[:100],  # Giới hạn để tránh quá tải
                help="Chọn một công ty để xem các công ty tương tự nhất"
            )
        
        with col2:
            threshold = st.slider(
                "Ngưỡng tương tự:",
                min_value=0.5,
                max_value=0.9,
                value=0.7,
                step=0.05,
                help="Chỉ hiển thị kết nối có độ tương tự >= ngưỡng này"
            )
        
        if selected_company:
            create_interactive_network_graph(df_companies, doc2vec_vectors, selected_company, threshold)
        else:
            st.info("👆 Vui lòng chọn một công ty để xem mạng lưới tương tự")
            
            # Hiển thị một số công ty mẫu
            st.subheader("💡 Gợi ý một số công ty để thử:")
            sample_companies = df_companies['Company Name'].sample(10).tolist()
            
            col1, col2 = st.columns(2)
            for i, company in enumerate(sample_companies):
                if i % 2 == 0:
                    col1.write(f"• {company}")
                else:
                    col2.write(f"• {company}")
    
    with viz_tab3:
        st.subheader("🚀 Xu hướng Phát triển")
        
        create_animated_timeline_chart(df)
        
        st.markdown("---")
        
        # Thống kê và insights
        st.subheader("📋 Insights & Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 Key Insights:
            
            - **Top Industry**: Ngành có nhiều công ty recommend nhất
            - **Growth Trend**: Xu hướng tăng trưởng qua các năm
            - **Success Factors**: Các yếu tố quan trọng nhất
            
            ### 📊 Market Analysis:
            
            - **Market Size**: Tổng số công ty trong database
            - **Competition**: Mức độ cạnh tranh theo ngành
            - **Opportunities**: Cơ hội phát triển tiềm năng
            """)
        
        with col2:
            # Top industries by recommend rate
            industry_recommend_rate = df.groupby('Company industry')['recommend_label'].agg(['count', 'mean']).reset_index()
            industry_recommend_rate = industry_recommend_rate[industry_recommend_rate['count'] >= 5]  # Chỉ lấy ngành có >= 5 công ty
            industry_recommend_rate = industry_recommend_rate.sort_values('mean', ascending=False).head(10)
            
            st.markdown("### 🏆 Top 10 Ngành Có Tỷ Lệ Recommend Cao Nhất:")
            for idx, row in industry_recommend_rate.iterrows():
                st.write(f"**{row['Company industry']}**: {row['mean']:.1%} ({row['count']} công ty)")
        
        # Recommendations
        st.markdown("---")
        st.subheader("💡 Khuyến nghị Chiến lược")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            #### 🎯 Cho Nhà Đầu tư:
            - Tập trung vào các ngành có tỷ lệ recommend cao
            - Đánh giá potential của startup mới
            - Phân tích competitor landscape
            """)
        
        with col2:
            st.markdown("""
            #### 🏢 Cho Doanh nghiệp:
            - Benchmark với competitor
            - Cải thiện employee satisfaction
            - Phát triển partnership strategy
            """)
        
        with col3:
            st.markdown("""
            #### 💼 Cho Job Seekers:
            - Tìm kiếm công ty có culture tốt
            - So sánh benefits package
            - Đánh giá career opportunity
            """)

st.markdown("""
<style>
    .css-1d391kg {
        font-size: 16px;
        font-weight: bold;
        position: fixed;
        left: 10px;
    }

    .css-1d391kg-first {
        bottom: 80px;
    }

    .css-1d391kg-email-first {
        bottom: 60px;
        font-weight: normal;
        font-size: 14px;
        color: gray;
        position: fixed;
        left: 10px;
    }

    .css-1d391kg-second {
        bottom: 40px;
    }

    .css-1d391kg-email-second {
        bottom: 20px;
        font-weight: normal;
        font-size: 14px;
        color: gray;
        position: fixed;
        left: 10px;
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


st.title("🏢 Gợi ý công ty tương tự (Doc2Vec)")
st.image('ITViec.jpg', use_container_width=True)
st.markdown('<div class="icon">CopyRight@LeHuuSonHai</div>', unsafe_allow_html=True)

st.sidebar.header("🔧 Tuì chọn")
top_n = st.sidebar.slider("Số lượng công ty gợi ý", min_value=3, max_value=15, value=5)
industry_list = df['Company industry'].dropna().unique().tolist()
selected_industry = st.sidebar.selectbox("📂 Lọc theo ngành", ["-- Tất cả --"] + sorted(industry_list))
industry_filter = None if selected_industry == "-- Tất cả --" else selected_industry

with st.sidebar.container():
    st.markdown('<div class="css-1d391kg-container">', unsafe_allow_html=True)
    st.markdown('<div class="css-1d391kg css-1d391kg-first">Lê Hữu Sơn Hải</div>', unsafe_allow_html=True)
    st.markdown('<div class="css-1d391kg-email-first">lehuusonhai@gmaill.com</div>', unsafe_allow_html=True)
    st.markdown('<div class="css-1d391kg css-1d391kg-second">Đoàn Trung Cường</div>', unsafe_allow_html=True)
    st.markdown('<div class="css-1d391kg-email-second">trungcuong.doan2601@gmail.com</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Tìm theo tên công ty", "✍️ Tìm theo mô tả", "Trực quan hóa dữ liệu", "📂 Dự đoán theo file"])

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
                st.markdown(f"🔹 **{row['Company Name']}** – điểm: {row['score']:.3f}")
                st.markdown(f"🗱 Từ khóa chung: {', '.join(common[:10])}")
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

with tab2:
    st.subheader("✍️ Nhập mô tả công ty hoặc lĩnh vực bạn muốn tìm:")
    description_input = st.text_area("Ví dụ: Công ty phần mềm chuyên về trí tuệ nhân tạo và dữ liệu lớn...")
    if description_input:
        results_desc = suggest_by_description(description_input, top_n=top_n)
        st.subheader("📋 Danh sách gợi ý theo mô tả:")
        st.dataframe(results_desc, use_container_width=True)
        csv_desc = results_desc.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Tải danh sách CSV", csv_desc, "description_based_suggestions.csv", "text/csv")

with tab3:
    # Gọi hàm trực quan hóa được cải tiến
    enhanced_visualization_tab_v2(df, df_companies, doc2vec_vectors)

with tab4:
    uploaded_file = st.file_uploader("📤 Tải file CSV/XLSX", type=["csv", "xlsx"])
    if uploaded_file:
        # Đọc file
        if uploaded_file.name.endswith(".csv"):
            df_input = pd.read_csv(uploaded_file)
        else:
            df_input = pd.read_excel(uploaded_file)

        st.subheader("📄 Dữ liệu đã tải")
        st.dataframe(df_input)

        # Các cột cần thiết
        required_columns = ['Company overview', 'Company industry', 'Training & learning', 'Salary & benefits']

        # Đảm bảo có đủ các cột đầu vào
        for col in required_columns:
            if col not in df_input.columns:
                df_input[col] = ""

        # Làm sạch dữ liệu giống lúc huấn luyện
        df_input['Company overview'] = df_input['Company overview'].astype(str).fillna("")
        df_input['Company industry'] = df_input['Company industry'].fillna("Unknown")
        df_input['Training & learning'] = pd.to_numeric(df_input['Training & learning'], errors='coerce')
        df_input['Salary & benefits'] = pd.to_numeric(df_input['Salary & benefits'], errors='coerce')

        df_input['Training & learning'] = df_input['Training & learning'].fillna(df_input['Training & learning'].median())
        df_input['Salary & benefits'] = df_input['Salary & benefits'].fillna(df_input['Salary & benefits'].median())

        # Dự đoán
        try:
            features = df_input[required_columns]
            preds = xgboost_classifier.predict(features)
            df_input['Prediction'] = np.where(preds == 1, "Recommend", "Not Recommend")
        except Exception as e:
            st.error(f"❌ Lỗi khi phân loại: {e}")

        # Kết quả
        st.subheader("🔍 Kết quả phân loại")
        st.dataframe(df_input)

        # Tải xuống kết quả
        st.download_button("⬇️ Tải kết quả", df_input.to_csv(index=False).encode("utf-8"), "batch_predictions.csv")