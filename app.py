import streamlit as st
import pandas as pd
import numpy as np
import ast
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(
    page_title="Movie Recommendation AI",
    page_icon="🎬",
    layout="wide"
)

# ======================================================
# LOAD DATA (SAFE)
# ======================================================
@st.cache_resource
def load_data():
    movie_path = "data_phim_full_images.csv"
    user_path = "danh_sach_nguoi_dung_moi.csv"

    if not os.path.exists(movie_path):
        st.error(f"❌ Không tìm thấy file {movie_path}")
        st.stop()

    if not os.path.exists(user_path):
        st.error(f"❌ Không tìm thấy file {user_path}")
        st.stop()

    movies = pd.read_csv(movie_path)
    users = pd.read_csv(user_path)

    return movies, users

movies_df, users_df = load_data()

# ======================================================
# PREPROCESS
# ======================================================
movies_df[['Đạo diễn', 'Thể loại phim', 'Mô tả']] = movies_df[
    ['Đạo diễn', 'Thể loại phim', 'Mô tả']
].fillna('')

movies_df['combined_features'] = (
    movies_df['Tên phim'] + " " +
    movies_df['Đạo diễn'] + " " +
    movies_df['Thể loại phim']
)

scaler = MinMaxScaler()
movies_df['popularity_scaled'] = scaler.fit_transform(
    movies_df[['Độ phổ biến']]
)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix)

users_df['history_list'] = users_df['5 phim coi gần nhất'].apply(
    lambda x: ast.literal_eval(x)[:5] if isinstance(x, str) else []
)

# ======================================================
# AI RECOMMENDATION
# ======================================================
def get_ai_recommendations(history_titles, top_k=50):
    indices = [
        movies_df[movies_df['Tên phim'] == t].index[0]
        for t in history_titles
        if not movies_df[movies_df['Tên phim'] == t].empty
    ]

    if not indices:
        return movies_df.sort_values(
            'Độ phổ biến', ascending=False
        ).head(top_k)

    sim_scores = np.mean(cosine_sim[indices], axis=0)
    pop_scores = movies_df['popularity_scaled'].values

    final_scores = 0.7 * sim_scores + 0.3 * pop_scores
    ranked = sorted(
        enumerate(final_scores),
        key=lambda x: x[1],
        reverse=True
    )

    rec_idx = [i for i, _ in ranked if i not in indices][:top_k]
    return movies_df.iloc[rec_idx]

def get_ai_new(history, excluded, k=10):
    recs = get_ai_recommendations(history, top_k=50)
    recs = recs[~recs['Tên phim'].isin(excluded)]
    return recs.head(k)

# ======================================================
# SESSION STATE
# ======================================================
if 'user' not in st.session_state:
    st.session_state.user = None
if 'shown' not in st.session_state:
    st.session_state.shown = []
if 'recs' not in st.session_state:
    st.session_state.recs = None

# ======================================================
# LOGIN
# ======================================================
st.title("🎬 Movie Recommendation AI")

if st.session_state.user is None:
    username = st.text_input("Nhập tên người dùng")

    if st.button("Đăng nhập"):
        row = users_df[users_df['Tên người dùng'] == username]
        if row.empty:
            st.error("❌ Người dùng không tồn tại")
        else:
            st.session_state.user = row.iloc[0]
            st.rerun()

else:
    st.success(f"Xin chào {st.session_state.user['Tên người dùng']}")

    history = st.session_state.user['history_list']
    st.write("🎞️ Lịch sử xem:", ", ".join(history))

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🎬 Đề xuất AI"):
            st.session_state.shown = []
            recs = get_ai_new(history, [])
            st.session_state.shown = recs['Tên phim'].tolist()
            st.session_state.recs = recs

    with col2:
        if st.button("🔄 Tạo mới"):
            recs = get_ai_new(history, st.session_state.shown)
            st.session_state.shown += recs['Tên phim'].tolist()
            st.session_state.recs = recs

    if st.session_state.recs is not None:
        st.markdown("---")
        cols = st.columns(5)
        for i, (_, r) in enumerate(st.session_state.recs.iterrows()):
            with cols[i % 5]:
                st.image(r['Link Poster'], use_container_width=True)
                st.caption(r['Tên phim'])

    if st.button("Đăng xuất"):
        st.session_state.clear()
        st.rerun()
