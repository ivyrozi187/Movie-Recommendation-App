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
# 1. CẤU HÌNH TRANG & CSS
# ==============================================================================
st.set_page_config(
    page_title="Movie RecSys AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stButton>button {
    width: 100%;
    border-radius: 5px;
    height: 3em;
    background-color: #ff4b4b;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. LOAD & XỬ LÝ DỮ LIỆU
# ==============================================================================
@st.cache_resource
def load_and_process_data():
    movies = pd.read_csv("data_phim_full_images.csv")
    users = pd.read_csv("danh_sach_nguoi_dung_gia_lap.csv")

    movies.fillna("", inplace=True)
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
        {g.strip() for s in movies['Thể loại phim'] for g in s.split(',')}
    )

    return movies, users, cosine_sim, all_genres

movies_df, users_df, cosine_sim, ALL_GENRES = load_and_process_data()

# ==============================================================================
# 3. HÀM GỢI Ý
# ==============================================================================
def get_ai_recommendations(history_titles, top_k=30, w_sim=0.7, w_pop=0.3):
    indices = []
    for title in history_titles:
        idx = movies_df[movies_df['Tên phim'] == title].index
        if not idx.empty:
            indices.append(idx[0])

    if not indices:
        return movies_df.sort_values(by='Độ phổ biến', ascending=False).head(top_k)

    sim_scores = np.mean(cosine_sim[indices], axis=0)
    final_scores = w_sim * sim_scores + w_pop * movies_df['popularity_scaled'].values

    scores = list(enumerate(final_scores))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    rec_idx = [i[0] for i in scores if i[0] not in indices][:top_k]

    return movies_df.iloc[rec_idx]

def get_genre_recommendations(genres, top_k=30):
    pattern = "|".join(genres)
    df = movies_df[movies_df['Thể loại phim'].str.contains(pattern, case=False, na=False)]
    return df.sort_values(by='Độ phổ biến', ascending=False).head(top_k)

def search_movie_func(query):
    return movies_df[movies_df['Tên phim'].str.contains(query, case=False, na=False)]

def draw_user_charts(history):
    genres = []
    for h in history:
        row = movies_df[movies_df['Tên phim'] == h]
        if not row.empty:
            genres += [g.strip() for g in row.iloc[0]['Thể loại phim'].split(',')]

    if not genres:
        st.warning("Chưa đủ dữ liệu thống kê")
        return

    df = pd.DataFrame(Counter(genres).items(), columns=['Thể loại', 'Số lần'])
    st.bar_chart(df.set_index('Thể loại'))

# ==============================================================================
# 4. SESSION STATE
# ==============================================================================
if 'user_mode' not in st.session_state:
    st.session_state.user_mode = None
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'user_genres' not in st.session_state:
    st.session_state.user_genres = []
if 'shown_movie_ids' not in st.session_state:
    st.session_state.shown_movie_ids = set()

# ==============================================================================
# 5. SIDEBAR
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

    elif st.session_state.user_mode in ['guest', 'register']:
        menu = st.radio("Chức năng", [
            "Đề xuất AI (Cơ bản)",
            "Theo Thể loại Đã chọn"
        ])
        if st.button("Thoát"):
            st.session_state.clear()
            st.rerun()
    else:
        menu = "LOGIN"

# ==============================================================================
# 6. LOGIN / REGISTER / GUEST
# ==============================================================================
if st.session_state.user_mode is None:
    t1, t2, t3 = st.tabs(["Đăng nhập", "Đăng ký", "Khách"])

    with t1:
        u = st.text_input("Tên đăng nhập")
        if st.button("Đăng nhập"):
            row = users_df[users_df['Tên người dùng'] == u]
            if not row.empty:
                st.session_state.user_mode = 'member'
                st.session_state.current_user = row.iloc[0]
                st.rerun()

    with t2:
        u = st.text_input("Tên mới")
        g = st.multiselect("Thể loại thích", ALL_GENRES)
        if st.button("Đăng ký"):
            if u and g:
                st.session_state.user_mode = 'register'
                st.session_state.user_genres = g
                st.rerun()

    with t3:
        g = st.multiselect("Chọn thể loại", ALL_GENRES)
        if st.button("Vào nhanh"):
            if g:
                st.session_state.user_mode = 'guest'
                st.session_state.user_genres = g
                st.rerun()

# ==============================================================================
# 7. MEMBER
# ==============================================================================
elif st.session_state.user_mode == 'member':
    user_history = st.session_state.current_user['history_list']

    # ===== ĐỀ XUẤT AI (THÊM GỢI Ý MỚI) =====
    if menu == "Đề xuất AI":
        st.header("🤖 Đề xuất AI")

        if st.button("🔄 Gợi ý mới – 10 phim khác", key="ai_new"):
            st.rerun()

        recs = get_ai_recommendations(user_history, top_k=30)
        recs = recs[~recs.index.isin(st.session_state.shown_movie_ids)].head(10)

        cols = st.columns(5)
        for i, (idx, r) in enumerate(recs.iterrows()):
            with cols[i % 5]:
                st.image(r['Link Poster'], use_container_width=True)
                st.caption(r['Tên phim'])

        st.session_state.shown_movie_ids.update(recs.index.tolist())

    # ===== TÌM KIẾM =====
    elif menu == "Tìm kiếm Phim":
        q = st.text_input("Nhập tên phim")
        if q:
            res = search_movie_func(q)
            if not res.empty:
                r = res.iloc[0]
                st.image(r['Link Poster'], width=300)
                st.write(r['Tên phim'])
                st.write(r['Thể loại phim'])
                st.write(r['Mô tả'])

    # ===== THEO THỂ LOẠI YÊU THÍCH (THÊM GỢI Ý MỚI) =====
    elif menu == "Theo Thể loại Yêu thích":
        fav = st.session_state.current_user['Phim yêu thích nhất']
        row = movies_df[movies_df['Tên phim'] == fav]

        if not row.empty:
            fav_genres = [g.strip() for g in row.iloc[0]['Thể loại phim'].split(',')]

            if st.button("🔄 Gợi ý mới – 10 phim khác", key="genre_new"):
                st.rerun()

            recs = get_genre_recommendations(fav_genres, top_k=30)
            recs = recs[~recs.index.isin(st.session_state.shown_movie_ids)].head(10)

            cols = st.columns(5)
            for i, (idx, r) in enumerate(recs.iterrows()):
                with cols[i % 5]:
                    st.image(r['Link Poster'], use_container_width=True)
                    st.caption(r['Tên phim'])

            st.session_state.shown_movie_ids.update(recs.index.tolist())

    # ===== THỐNG KÊ =====
    elif menu == "Thống kê Cá nhân":
        draw_user_charts(user_history)

# ==============================================================================
# 8. GUEST / REGISTER
# ==============================================================================
else:
    genres = st.session_state.user_genres

    recs = get_genre_recommendations(genres, top_k=10)
    cols = st.columns(5)
    for i, (_, r) in enumerate(recs.iterrows()):
        with cols[i % 5]:
            st.image(r['Link Poster'], use_container_width=True)
            st.caption(r['Tên phim'])
