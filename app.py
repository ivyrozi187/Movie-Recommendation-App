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
# 2. LOAD & PREPROCESS DATA
# ==============================================================================
@st.cache_resource
def load_and_process_data():
    movies = pd.read_csv("data_phim_full_images.csv")
    users = pd.read_csv("danh_sach_nguoi_dung_moi.csv")

    movies[['Đạo diễn', 'Thể loại phim', 'Mô tả']] = movies[
        ['Đạo diễn', 'Thể loại phim', 'Mô tả']
    ].fillna('')

    movies['combined_features'] = (
        movies['Tên phim'] + " " +
        movies['Đạo diễn'] + " " +
        movies['Thể loại phim']
    )

    scaler = MinMaxScaler()
    movies['popularity_scaled'] = scaler.fit_transform(movies[['Độ phổ biến']])

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix)

    users['history_list'] = users['5 phim coi gần nhất'].apply(
        lambda x: ast.literal_eval(x)[:5] if isinstance(x, str) else []
    )

    all_genres = sorted({
        g.strip()
        for genres in movies['Thể loại phim']
        for g in genres.split(',')
    })

    return movies, users, cosine_sim, all_genres

movies_df, users_df, cosine_sim, ALL_GENRES = load_and_process_data()

# ==============================================================================
# 3. CORE FUNCTIONS
# ==============================================================================
def get_ai_recommendations(history_titles, top_k=50, w_sim=0.7, w_pop=0.3):
    indices = [
        movies_df[movies_df['Tên phim'] == t].index[0]
        for t in history_titles
        if not movies_df[movies_df['Tên phim'] == t].empty
    ]

    if not indices:
        return movies_df.sort_values('Độ phổ biến', ascending=False).head(top_k)

    sim_scores = np.mean(cosine_sim[indices], axis=0)
    pop_scores = movies_df['popularity_scaled'].values
    final_scores = w_sim * sim_scores + w_pop * pop_scores

    ranked = sorted(enumerate(final_scores), key=lambda x: x[1], reverse=True)
    rec_idx = [i for i, _ in ranked if i not in indices][:top_k]

    return movies_df.iloc[rec_idx]

def get_ai_recommendations_new(history_titles, exclude_titles, top_k=10):
    recs = get_ai_recommendations(history_titles, top_k=50)
    if exclude_titles:
        recs = recs[~recs['Tên phim'].isin(exclude_titles)]
    return recs.head(top_k)

# ==============================================================================
# 4. SESSION STATE
# ==============================================================================
if 'user_mode' not in st.session_state:
    st.session_state.user_mode = None
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'ai_shown_movies' not in st.session_state:
    st.session_state.ai_shown_movies = []
if 'ai_recs' not in st.session_state:
    st.session_state.ai_recs = None

# ==============================================================================
# 5. SIDEBAR
# ==============================================================================
with st.sidebar:
    st.title("🎬 DreamStream")
    if st.session_state.user_mode == 'member':
        menu = st.radio(
            "Chức năng",
            ["Đề xuất AI", "Thống kê Cá nhân"]
        )
        if st.button("Đăng xuất"):
            st.session_state.clear()
            st.rerun()
    else:
        menu = "Login"

# ==============================================================================
# 6. LOGIN
# ==============================================================================
if st.session_state.user_mode is None:
    st.title("Đăng nhập")
    u = st.text_input("Tên người dùng")
    if st.button("Đăng nhập"):
        row = users_df[users_df['Tên người dùng'] == u]
        if not row.empty:
            st.session_state.user_mode = 'member'
            st.session_state.current_user = row.iloc[0]
            st.rerun()
        else:
            st.error("Người dùng không tồn tại")

# ==============================================================================
# 7. MEMBER – ĐỀ XUẤT AI (CÓ TẠO MỚI)
# ==============================================================================
elif st.session_state.user_mode == 'member':
    user_history = st.session_state.current_user.get('history_list', [])

    if menu == "Đề xuất AI":
        st.header("🤖 Đề xuất Phim Thông minh")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🎬 Đề xuất AI"):
                st.session_state.ai_shown_movies = []
                recs = get_ai_recommendations_new(user_history, [])
                st.session_state.ai_shown_movies = recs['Tên phim'].tolist()
                st.session_state.ai_recs = recs

        with col2:
            if st.button("🔄 Tạo mới"):
                recs = get_ai_recommendations_new(
                    user_history,
                    st.session_state.ai_shown_movies
                )
                st.session_state.ai_shown_movies += recs['Tên phim'].tolist()
                st.session_state.ai_recs = recs

        if st.session_state.ai_recs is not None:
            st.markdown("---")
            cols = st.columns(5)
            for i, (idx, row) in enumerate(st.session_state.ai_recs.iterrows()):
                with cols[i % 5]:
                    st.image(row['Link Poster'], use_container_width=True)
                    st.caption(f"**{row['Tên phim']}**")
                    with st.expander("Chi tiết"):
                        st.write(f"⭐ {row['Độ phổ biến']:.1f}")
                        st.write(f"🎭 {row['Thể loại phim']}")
