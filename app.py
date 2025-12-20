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
    layout="wide"
)

# ==============================================================================
# 2. LOAD DATA
# ==============================================================================
@st.cache_resource
def load_data():
    movies = pd.read_csv("data_phim_full_images.csv")
    users = pd.read_csv("danh_sach_nguoi_dung_gia_lap.csv")

    movies.fillna('', inplace=True)

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

    users['history_list'] = users['5 phim coi gần nhất'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )

    all_genres = sorted(
        {g.strip() for row in movies['Thể loại phim'] for g in row.split(',')}
    )

    return movies, users, cosine_sim, all_genres

movies_df, users_df, cosine_sim, ALL_GENRES = load_data()

# ==============================================================================
# 3. SESSION STATE (KHỞI TẠO ĐÚNG)
# ==============================================================================
if 'user_mode' not in st.session_state:
    st.session_state.user_mode = None

if 'current_user' not in st.session_state:
    st.session_state.current_user = None

if 'user_genres' not in st.session_state:
    st.session_state.user_genres = []

if 'ai_seen' not in st.session_state:
    st.session_state.ai_seen = set()

if 'search_seen' not in st.session_state:
    st.session_state.search_seen = set()

if 'genre_seen' not in st.session_state:
    st.session_state.genre_seen = set()

# ==============================================================================
# 4. RECOMMENDATION FUNCTIONS (FIX BUG)
# ==============================================================================
def get_ai_recommendations(history_titles, top_k=10, w_sim=0.7, w_pop=0.3, exclude=None):
    if exclude is None:
        exclude = set()

    indices = []
    for t in history_titles:
        idx = movies_df[movies_df['Tên phim'] == t].index
        if not idx.empty:
            indices.append(idx[0])

    sim_scores = np.mean(cosine_sim[indices], axis=0) if indices else np.zeros(len(movies_df))
    final_scores = w_sim * sim_scores + w_pop * movies_df['popularity_scaled'].values

    ranked = sorted(enumerate(final_scores), key=lambda x: x[1], reverse=True)

    rec_idx = [
        i for i, _ in ranked
        if i not in indices and i not in exclude
    ][:top_k]

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


def search_movie(query):
    return movies_df[movies_df['Tên phim'].str.contains(query, case=False)]

# ==============================================================================
# 5. SIDEBAR
# ==============================================================================
with st.sidebar:
    st.title("🎬 DreamStream")

    if st.session_state.user_mode == 'member':
        menu = st.radio("Chức năng", ["Đề xuất AI", "Tìm kiếm", "Theo thể loại"])
        if st.button("Đăng xuất"):
            for k in ['ai_seen', 'search_seen', 'genre_seen']:
                st.session_state[k].clear()
            st.session_state.user_mode = None
            st.session_state.current_user = None
            st.rerun()

    elif st.session_state.user_mode in ['guest', 'register']:
        menu = st.radio("Chức năng", ["Đề xuất AI", "Theo thể loại"])
        if st.button("Thoát"):
            for k in ['ai_seen', 'search_seen', 'genre_seen']:
                st.session_state[k].clear()
            st.session_state.user_mode = None
            st.session_state.user_genres = []
            st.rerun()
    else:
        menu = "Login"

# ==============================================================================
# 6. LOGIN / REGISTER / GUEST
# ==============================================================================
if st.session_state.user_mode is None:
    tab1, tab2, tab3 = st.tabs(["Đăng nhập", "Đăng ký", "Khách"])

    with tab1:
        u = st.text_input("Tên đăng nhập")
        if st.button("Đăng nhập"):
            r = users_df[users_df['Tên người dùng'] == u]
            if not r.empty:
                st.session_state.user_mode = 'member'
                st.session_state.current_user = r.iloc[0]
                st.rerun()

    with tab2:
        u = st.text_input("Tên mới")
        g = st.multiselect("Thể loại thích", ALL_GENRES)
        if st.button("Đăng ký"):
            st.session_state.user_mode = 'register'
            st.session_state.user_genres = g
            st.rerun()

    with tab3:
        g = st.multiselect("Chọn thể loại", ALL_GENRES)
        if st.button("Vào ngay"):
            st.session_state.user_mode = 'guest'
            st.session_state.user_genres = g
            st.rerun()

# ==============================================================================
# 7. MEMBER
# ==============================================================================
elif st.session_state.user_mode == 'member':
    history = st.session_state.current_user['history_list']

    if menu == "Đề xuất AI":
        if st.button("🔄 Tạo mới"):
            st.session_state.ai_seen.clear()

        recs, idxs = get_ai_recommendations(history, exclude=st.session_state.ai_seen)
        st.session_state.ai_seen.update(idxs)

    elif menu == "Tìm kiếm":
        q = st.text_input("Nhập tên phim")
        if q:
            m = search_movie(q)
            if not m.empty:
                if st.button("🔄 Phim tương tự khác"):
                    st.session_state.search_seen.clear()

                recs, idxs = get_ai_recommendations(
                    [m.iloc[0]['Tên phim']], w_sim=1, w_pop=0,
                    exclude=st.session_state.search_seen
                )
                st.session_state.search_seen.update(idxs)

    elif menu == "Theo thể loại":
        fav = st.session_state.current_user['Phim yêu thích nhất']
        row = movies_df[movies_df['Tên phim'] == fav]
        if not row.empty:
            genres = [g.strip() for g in row.iloc[0]['Thể loại phim'].split(',')]
            if st.button("🔄 Tạo mới theo thể loại"):
                st.session_state.genre_seen.clear()

            recs, idxs = get_genre_recommendations(genres, exclude=st.session_state.genre_seen)
            st.session_state.genre_seen.update(idxs)

# ==============================================================================
# 8. GUEST / REGISTER
# ==============================================================================
elif st.session_state.user_mode in ['guest', 'register']:
    genres = st.session_state.user_genres

    if menu == "Đề xuất AI":
        if st.button("🔄 Tạo mới"):
            st.session_state.genre_seen.clear()

        recs, idxs = get_genre_recommendations(genres, exclude=st.session_state.genre_seen)
        st.session_state.genre_seen.update(idxs)

    elif menu == "Theo thể loại":
        g = st.selectbox("Chọn thể loại", genres)
        if st.button("🔄 Tạo mới"):
            st.session_state.genre_seen.clear()

        recs, idxs = get_genre_recommendations([g], exclude=st.session_state.genre_seen)
        st.session_state.genre_seen.update(idxs)
