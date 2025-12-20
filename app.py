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
import random

# ==============================================================================
# CONFIG
# ==============================================================================
st.set_page_config(
    page_title="DreamStream – Movie Recommendation AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# LOAD DATA (SAFE)
# ==============================================================================
@st.cache_resource
def load_and_process_data():
    movies = pd.read_csv("data_phim_full_images.csv")

    try:
        users = pd.read_csv("danh_sach_nguoi_dung_moi.csv")
    except:
        users = pd.DataFrame(columns=[
            "Tên người dùng",
            "5 phim coi gần nhất",
            "Phim yêu thích nhất"
        ])

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

    if '5 phim coi gần nhất' in users.columns:
        users['history_list'] = users['5 phim coi gần nhất'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else []
        )
    else:
        users['history_list'] = []

    users['Tên người dùng'] = users['Tên người dùng'].astype(str).str.strip().str.lower()

    all_genres = sorted(
        {g.strip() for row in movies['Thể loại phim'] for g in row.split(',')}
    )

    return movies, users, cosine_sim, all_genres


movies_df, users_df, cosine_sim, ALL_GENRES = load_and_process_data()

# ==============================================================================
# SESSION STATE
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
# RECOMMENDATION FUNCTIONS (FIX TẠO MỚI)
# ==============================================================================
def get_ai_recommendations(history_titles, top_k=10, exclude=None):
    if exclude is None:
        exclude = set()

    watched = []
    for t in history_titles:
        idx = movies_df[movies_df['Tên phim'] == t].index
        if not idx.empty:
            watched.append(idx[0])

    sim_scores = np.mean(cosine_sim[watched], axis=0) if watched else np.zeros(len(movies_df))
    scores = 0.7 * sim_scores + 0.3 * movies_df['popularity_scaled'].values

    ranked = list(enumerate(scores))
    ranked = [x for x in ranked if x[0] not in watched and x[0] not in exclude]
    ranked = sorted(ranked, key=lambda x: x[1], reverse=True)[:50]

    random.shuffle(ranked)
    rec_idx = [i for i, _ in ranked[:top_k]]

    return movies_df.iloc[rec_idx], rec_idx


def get_genre_recommendations(genres, top_k=10, exclude=None):
    if exclude is None:
        exclude = set()
    if not genres:
        return pd.DataFrame(), []

    df = movies_df[movies_df['Thể loại phim'].str.contains('|'.join(genres), case=False)]
    df = df[~df.index.isin(exclude)]
    top = df.sort_values(by='Độ phổ biến', ascending=False).head(50)

    idxs = list(top.index)
    random.shuffle(idxs)
    rec_idx = idxs[:top_k]

    return movies_df.loc[rec_idx], rec_idx


def search_movie_func(q):
    return movies_df[movies_df['Tên phim'].str.contains(q, case=False)]


def draw_user_charts(history):
    genres = []
    for t in history:
        r = movies_df[movies_df['Tên phim'] == t]
        if not r.empty:
            genres += [g.strip() for g in r.iloc[0]['Thể loại phim'].split(',')]

    if not genres:
        st.warning("Chưa có dữ liệu")
        return

    df = pd.Series(genres).value_counts().reset_index()
    df.columns = ['Thể loại', 'Số phim']

    fig, ax = plt.subplots()
    sns.barplot(data=df, x='Số phim', y='Thể loại', ax=ax)
    st.pyplot(fig)

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.title("🎬 DreamStream")

    if st.session_state.user_mode == 'member':
        menu = st.radio("Chức năng", [
            "Đề xuất AI",
            "Tìm kiếm Phim",
            "Theo Thể loại Yêu thích",
            "Thống kê Cá nhân"
        ])
        if st.button("Đăng xuất"):
            st.session_state.clear()
            st.rerun()

    elif st.session_state.user_mode == 'guest':
        menu = st.radio("Chức năng", ["Đề xuất AI (Cơ bản)", "Theo Thể loại Đã chọn"])
        if st.button("Thoát"):
            st.session_state.clear()
            st.rerun()
    else:
        menu = "Login"

# ==============================================================================
# LOGIN / REGISTER / GUEST (FIX LOGIN)
# ==============================================================================
if st.session_state.user_mode is None:
    tab1, tab2, tab3 = st.tabs(["Đăng nhập", "Đăng ký", "Khách"])

    with tab1:
        u = st.text_input("Tên đăng nhập")
        if st.button("Đăng nhập"):
            u_clean = u.strip().lower()
            r = users_df[users_df['Tên người dùng'] == u_clean]
            if not r.empty:
                st.session_state.user_mode = 'member'
                st.session_state.current_user = r.iloc[0]
                st.rerun()
            else:
                st.error("❌ Không tìm thấy người dùng")

    with tab2:
        u = st.text_input("Tên mới")
        g = st.multiselect("Thể loại yêu thích", ALL_GENRES)
        if st.button("Đăng ký & Vào ngay"):
            st.session_state.user_mode = 'member'
            st.session_state.current_user = {
                'Tên người dùng': u,
                'history_list': [],
                'Phim yêu thích nhất': ''
            }
            st.session_state.user_genres = g
            st.rerun()

    with tab3:
        g = st.multiselect("Chọn thể loại", ALL_GENRES)
        if st.button("Vào ngay"):
            st.session_state.user_mode = 'guest'
            st.session_state.user_genres = g
            st.rerun()

# ==============================================================================
# MEMBER
# ==============================================================================
elif st.session_state.user_mode == 'member':
    history = st.session_state.current_user.get('history_list', [])

    if menu == "Đề xuất AI":
        if st.button("🔄 Tạo mới"):
            st.session_state.ai_seen.clear()

        recs, idxs = get_ai_recommendations(history, exclude=st.session_state.ai_seen)
        st.session_state.ai_seen.update(idxs)

    elif menu == "Tìm kiếm Phim":
        q = st.text_input("Tên phim")
        if q:
            r = search_movie_func(q)
            if not r.empty:
                m = r.iloc[0]
                if st.button("🔄 Phim tương tự khác"):
                    st.session_state.search_seen.clear()

                recs, idxs = get_ai_recommendations(
                    [m['Tên phim']], exclude=st.session_state.search_seen
                )
                st.session_state.search_seen.update(idxs)

    elif menu == "Theo Thể loại Yêu thích":
        fav = st.session_state.current_user.get('Phim yêu thích nhất', '')
        if fav:
            row = movies_df[movies_df['Tên phim'] == fav]
            if not row.empty:
                genres = [g.strip() for g in row.iloc[0]['Thể loại phim'].split(',')]
                if st.button("🔄 Tạo mới"):
                    st.session_state.genre_seen.clear()
                recs, idxs = get_genre_recommendations(genres, exclude=st.session_state.genre_seen)
                st.session_state.genre_seen.update(idxs)

    elif menu == "Thống kê Cá nhân":
        draw_user_charts(history)

# ==============================================================================
# GUEST
# ==============================================================================
elif st.session_state.user_mode == 'guest':
    if st.button("🔄 Tạo mới"):
        st.session_state.genre_seen.clear()

    recs, idxs = get_genre_recommendations(
        st.session_state.user_genres, exclude=st.session_state.genre_seen
    )
    st.session_state.genre_seen.update(idxs)
