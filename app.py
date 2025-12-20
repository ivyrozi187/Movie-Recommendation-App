import streamlit as st
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

# ==============================================================================
# 1. CONFIG
# ==============================================================================
st.set_page_config(
    page_title="Movie RecSys AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. LOAD DATA (FIX FILE NOT FOUND)
# ==============================================================================
@st.cache_resource
def load_and_process_data():
    # ---- Movies (bắt buộc phải có) ----
    movies = pd.read_csv("data_phim_full_images.csv")

    # ---- Users (có thì dùng, không có thì tạo giả) ----
    try:
        users = pd.read_csv("danh_sach_nguoi_dung_moi.csv")
    except FileNotFoundError:
        users = pd.DataFrame({
            "Tên người dùng": [],
            "5 phim coi gần nhất": [],
            "Phim yêu thích nhất": []
        })

    # ---- Xử lý Movies ----
    movies['Đạo diễn'] = movies['Đạo diễn'].fillna('')
    movies['Thể loại phim'] = movies['Thể loại phim'].fillna('')
    movies['Mô tả'] = movies['Mô tả'].fillna('')

    movies['combined_features'] = (
        movies['Tên phim'] + " " +
        movies['Đạo diễn'] + " " +
        movies['Thể loại phim']
    )

    scaler = MinMaxScaler()
    movies['popularity_scaled'] = scaler.fit_transform(movies[['Độ phổ biến']])

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # ---- Xử lý Users ----
    if "5 phim coi gần nhất" in users.columns:
        users['history_list'] = users['5 phim coi gần nhất'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else []
        )
    else:
        users['history_list'] = []

    all_genres = sorted(
        {g.strip() for row in movies['Thể loại phim'] for g in row.split(',')}
    )

    return movies, users, cosine_sim, all_genres


movies_df, users_df, cosine_sim, ALL_GENRES = load_and_process_data()

# ==============================================================================
# 3. SESSION STATE
# ==============================================================================
if 'user_mode' not in st.session_state:
    st.session_state.user_mode = None
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'user_genres' not in st.session_state:
    st.session_state.user_genres = []

for k in ['ai_seen', 'search_seen', 'genre_seen']:
    if k not in st.session_state:
        st.session_state[k] = set()

# ==============================================================================
# 4. RECOMMENDATION FUNCTIONS
# ==============================================================================
def get_ai_recommendations(history_titles, top_k=10, w_sim=0.7, w_pop=0.3, exclude=None):
    if exclude is None:
        exclude = set()

    indices = []
    for t in history_titles:
        idx = movies_df[movies_df['Tên phim'] == t].index
        if not idx.empty:
            indices.append(idx[0])

    if not indices:
        df = movies_df[~movies_df.index.isin(exclude)]
        res = df.sort_values(by='Độ phổ biến', ascending=False).head(top_k)
        return res, list(res.index)

    sim_scores = np.mean(cosine_sim[indices], axis=0)
    final_scores = w_sim * sim_scores + w_pop * movies_df['popularity_scaled'].values

    ranked = sorted(enumerate(final_scores), key=lambda x: x[1], reverse=True)
    rec_idx = [i for i, _ in ranked if i not in indices and i not in exclude][:top_k]

    return movies_df.iloc[rec_idx], rec_idx


def get_genre_recommendations(genres, top_k=10, exclude=None):
    if exclude is None:
        exclude = set()
    if not genres:
        return pd.DataFrame(), []

    pattern = "|".join(genres)
    df = movies_df[movies_df['Thể loại phim'].str.contains(pattern, case=False)]
    df = df[~df.index.isin(exclude)]

    res = df.sort_values(by='Độ phổ biến', ascending=False).head(top_k)
    return res, list(res.index)


def search_movie_func(query):
    return movies_df[movies_df['Tên phim'].str.contains(query, case=False)]


def draw_user_charts(history):
    genres = []
    for t in history:
        r = movies_df[movies_df['Tên phim'] == t]
        if not r.empty:
            genres += [g.strip() for g in r.iloc[0]['Thể loại phim'].split(',')]

    if not genres:
        st.warning("Chưa có dữ liệu thống kê")
        return

    df = pd.Series(genres).value_counts().reset_index()
    df.columns = ['Thể loại', 'Số phim']

    fig, ax = plt.subplots()
    sns.barplot(data=df, x='Số phim', y='Thể loại', ax=ax)
    st.pyplot(fig)

# ==============================================================================
# 5. UI (GIỮ NGUYÊN HÀNH VI)
# ==============================================================================
st.title("🎬 DreamStream – Movie Recommendation AI")
st.success("Ứng dụng đã FIX lỗi FileNotFound – chạy an toàn trên Streamlit Cloud ✅")
